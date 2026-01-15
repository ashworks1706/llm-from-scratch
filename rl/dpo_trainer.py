# we have only one model but we use it in two ways :
# policy model is the model we're trianing, we inject lora adapters in it so it gets updated 
# the anchor in KL divergence -> refrernce model is the same model but we dont attach lora adapters to it, its just frozen its just original SFT model

# there's no generation in RL methods, the model is just going through forward pass we proivde the model with both instruction+question and chosen response with rejected response
# from the dataset, and then the model just predicts the sequence of the next token suhc that how likely is this specific response instead of generation
# we basically perfomr forward pass on both sequences of chosen and rejected
# we mask the prompt and only get log_prob of responses
# reference model which is basically the same but frozen model also evaluates both and proposes its own lob prob of responeses of chosen and rejected
# so now we have log_prob_chosen, log_prob_rejected and ref_log_prob_chosen, ref_log_prob_rejected 
# then we calculate loss by subtracting both chosen and rejected for policy and reference log probs 
# so DPO loss aims to maximize policy diff but penalize if we drift too far from reference 
# so its logits = beta * (policy_diff - reference_diff)
# loss = -log_sigmoid(logtis)
# so if policy_diff > reference_diff: we prefer chosen more than reference that its doing good 
# if policy_diff < reference_diff : we're getting worse 
# basically encouring "like what SFT liked, but prefer chosen over rejected more"
# the model outputs logits for every vocab token at every position in forward pass and we convert them to probaiblities (softmax)
# then we extract probability of specific tokens that actually appeared 

# why do we need BOTH chosen AND rejected?
# SFT trains on chosen only: makes P(chosen) higher but doesn't know about rejected
# DPO trains on BOTH: makes P(chosen) higher AND P(rejected) lower simultaneously
# this WIDENS the gap between good and bad responses!
# example: after SFT P(chosen)=0.6, P(rejected)=0.4 (gap=0.2)
#          after DPO P(chosen)=0.9, P(rejected)=0.1 (gap=0.8)
# DPO explicitly teaches: "this is good, that is bad" not just "generate this"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama3.model import Llama
from utils.config import Config
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lora'))
from inject_lora import inject_lora_to_model, save_lora_adapters
from dpo_dataset import DPODataset


class DPOTrainer:
    
    def __init__(self, config, train_dataset_path, sft_checkpoint, lora_rank=16, lora_alpha=32, beta=0.1):
        self.config = config
        self.beta = beta  # temperature for DPO loss, controls strength of preference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load DPO dataset with preference pairs
        self.dataset = DPODataset(train_dataset_path, config.max_sequence_length)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        # create policy model (the one being trained)
        self.policy_model = Llama(config).to(self.device)
        checkpoint = torch.load(sft_checkpoint, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded SFT checkpoint for policy model")
        
        # inject LoRA into policy - only these adapters will be trained
        # base weights stay frozen, only A and B matrices get gradients
        inject_lora_to_model(self.policy_model, rank=lora_rank, alpha=lora_alpha)
        
        # create reference model (frozen SFT baseline)
        # same architecture, same weights, but NO LoRA and completely frozen
        # this prevents policy from drifting into crazy territory
        self.reference_model = Llama(config).to(self.device)
        self.reference_model.load_state_dict(checkpoint['model_state_dict'])
        self.reference_model.eval()  # eval mode: disables dropout
        for param in self.reference_model.parameters():
            param.requires_grad = False  # no gradients = frozen
        print("Reference model loaded and frozen")
        
        # optimizer only trains LoRA parameters in policy
        # reference stays untouched forever
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.policy_model.parameters()),
            lr=config.learning_rate
        )
    
    def compute_log_probs(self, model, input_ids, mask):
        # computes: how likely is this specific sequence?
        # NOT generation! we're scoring existing tokens from dataset
        
        # forward pass to get logits (raw scores)
        # input_ids: (batch, seq_len)
        # logits: (batch, seq_len, vocab_size)
        logits = model(input_ids)
        
        # shift for next-token prediction (just like pretraining/SFT)
        # logits[:, :-1] predicts input_ids[:, 1:]
        # example: position 0 predicts token at position 1
        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        mask = mask[:, 1:].contiguous()
        
        # convert logits to log probabilities
        # log_softmax converts raw scores to log(P(token))
        # shape stays: (batch, seq_len-1, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # gather log prob of ACTUAL token at each position
        # labels tell us which token actually appeared
        # we extract P(that specific token) not all 50k vocab probs
        # example: if label[0]=42, we get log_probs[0, 42]
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        # result: (batch, seq_len-1) - one log prob per position
        
        # apply mask and sum over sequence
        # mask=0 for prompt tokens (ignore them)
        # mask=1 for response tokens (include them)
        # we only care about response quality, not prompt prediction
        masked_log_probs = gathered_log_probs * mask
        
        # sum to get total log probability of the response
        # this single number is our "score" for this sequence
        # higher = model thinks this sequence is more likely
        sequence_log_prob = masked_log_probs.sum(dim=-1)
        
        return sequence_log_prob  # shape: (batch,)
    
    def train_step(self, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
        # one training iteration with a batch of preference pairs
        # we compute log probs for both chosen and rejected
        # then use DPO loss to make chosen more likely than rejected
        
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        chosen_mask = chosen_mask.to(self.device)
        rejected_mask = rejected_mask.to(self.device)
        
        self.optimizer.zero_grad()
        
        # compute policy log probs (model being trained)
        # these forward passes TRACK GRADIENTS (needed for backward)
        policy_chosen_log_prob = self.compute_log_probs(self.policy_model, chosen_ids, chosen_mask)
        policy_rejected_log_prob = self.compute_log_probs(self.policy_model, rejected_ids, rejected_mask)
        
        # compute reference log probs (frozen SFT model)
        # no_grad because reference never trains, saves memory
        with torch.no_grad():
            ref_chosen_log_prob = self.compute_log_probs(self.reference_model, chosen_ids, chosen_mask)
            ref_rejected_log_prob = self.compute_log_probs(self.reference_model, rejected_ids, rejected_mask)
        
        # DPO loss calculation
        # how much does policy prefer chosen over rejected?
        # this is log(P(chosen)/P(rejected)) = log(P(chosen)) - log(P(rejected))
        policy_log_ratio = policy_chosen_log_prob - policy_rejected_log_prob
        
        # how much did reference prefer chosen over rejected?
        ref_log_ratio = ref_chosen_log_prob - ref_rejected_log_prob
        
        # we want policy to prefer chosen MORE than reference did
        # but not drift too far (KL penalty is implicit in this formula)
        # beta controls strength: higher beta = stronger preference enforcement
        logits = self.beta * (policy_log_ratio - ref_log_ratio)
        
        # final DPO loss: negative log sigmoid
        # when logits is large (policy prefers chosen much more) → loss is low ✓
        # when logits is small (policy doesn't prefer chosen enough) → loss is high ✗
        # logsigmoid is numerically stable version of log(sigmoid(x))
        loss = -F.logsigmoid(logits).mean()
        
        # backward pass: compute gradients for policy model
        # only LoRA adapters get gradients, base weights stay frozen
        loss.backward()
        
        # optimizer step: update LoRA weights based on gradients
        self.optimizer.step()
        
        # compute metrics for monitoring
        with torch.no_grad():
            # accuracy: how often does policy prefer chosen over rejected?
            # if policy_log_ratio > 0, chosen is more likely than rejected
            accuracy = (policy_log_ratio > 0).float().mean()
            
            # reward margin: how much better is policy than reference?
            # positive means policy is improving over reference
            reward_margin = (policy_log_ratio - ref_log_ratio).mean()
        
        return loss.item(), accuracy.item(), reward_margin.item()
    
    def train_epoch(self, epoch):
        self.policy_model.train()  # enable training mode
        total_loss = 0
        total_accuracy = 0
        total_reward_margin = 0
        
        # loop over all preference pairs in dataset
        for batch_idx, (chosen_ids, rejected_ids, chosen_mask, rejected_mask) in enumerate(self.dataloader):
            loss, accuracy, reward_margin = self.train_step(
                chosen_ids, rejected_ids, chosen_mask, rejected_mask
            )
            
            total_loss += loss
            total_accuracy += accuracy
            total_reward_margin += reward_margin
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}")
                print(f"  Loss: {loss:.4f}, Acc: {accuracy:.4f}, Margin: {reward_margin:.4f}")
        
        # calculate averages for this epoch
        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = total_accuracy / len(self.dataloader)
        avg_reward_margin = total_reward_margin / len(self.dataloader)
        
        return avg_loss, avg_accuracy, avg_reward_margin
    
    def train(self):
        print("Starting DPO training...")
        os.makedirs("dpo_checkpoints", exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            avg_loss, avg_accuracy, avg_reward_margin = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch} summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.4f} (% policy prefers chosen)")
            print(f"  Reward Margin: {avg_reward_margin:.4f} (improvement over reference)")
            
            # save LoRA adapters only (not full model)
            adapter_path = f"dpo_checkpoints/dpo_lora_epoch_{epoch+1}.pt"
            save_lora_adapters(self.policy_model, adapter_path)
            print(f"  Saved to {adapter_path}\n")
        
        print("DPO training complete!")
        print("Model now prefers high-quality responses and avoids poor ones!")


if __name__ == "__main__":
    config = Config(
        model_name="llama3",
        version="dpo",
        max_sequence_length=512,
        embedding_size=512,
        num_attention_heads=8,
        num_layers=4,
        dropout_rate=0.1,
        learning_rate=5e-6,  # very low for RL
        batch_size=4,
        num_epochs=3,
        vocab_size=128000,
        tokenizer_type="tiktoken",
        num_kv_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=500000.0
    )
    
