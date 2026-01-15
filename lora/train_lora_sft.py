# sft with LoRA: parameter-efficient fine-tuning
# instead of training ALL 7B parameters, we only train ~10M LoRA adapters
# base model stays frozen, only A and B matrices learn
# this means: less memory, faster training, can run on smaller GPUs


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama3.model import Llama
from utils.config import Config
from inject_lora import inject_lora_to_model, count_parameters, save_lora_adapters
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sft'))
from dataset import SFTDataset, SFTDataPreprocessor


class LoRASFTTrainer:
   
    def __init__(self, config, train_dataset_path, pretrained_checkpoint=None, lora_rank=16, lora_alpha=32):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.dataset = SFTDataset(train_dataset_path, config.max_sequence_length)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        self.model = Llama(config).to(self.device)
        
        if pretrained_checkpoint:
            print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Pretrained weights loaded!")
        
        # THIS IS THE KEY DIFFERENCE: inject LoRA into the model
        # this wraps wq, wk, wv, wo with LoRA layers
        # original weights get frozen, only LoRA adapters (A, B) are trainable
        print("Injecting LoRA adapters into model...")
        inject_lora_to_model(self.model, rank=lora_rank, alpha=lora_alpha)
        
        # optimizer now only trains LoRA parameters!
        # model.parameters() returns ALL params, but only LoRA ones have requires_grad=True
        # so optimizer only updates those
        # we can use HIGHER learning rate than full SFT because we're training fewer params
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),  # only trainable params
            lr=config.learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def train_step(self, inputs, targets, loss_mask):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        loss_mask = loss_mask.to(self.device)
        
        self.optimizer.zero_grad()
        
        logits = self.model(inputs)
        
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        loss_mask = loss_mask.view(-1)
        
        loss_per_position = self.criterion(logits, targets)
        
        # masked loss: only train on response tokens
        masked_loss = loss_per_position * loss_mask
        loss = masked_loss.sum() / loss_mask.sum()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets, loss_mask) in enumerate(self.dataloader):
            loss = self.train_step(inputs, targets, loss_mask)
            total_loss += loss
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss
    
    def train(self):
        print("Starting LoRA SFT training...")
        os.makedirs("lora_checkpoints", exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
            
            # save LoRA adapters (tiny files, few MB!)
            # we DON'T save the full model because base weights are frozen
            adapter_path = f"lora_checkpoints/lora_adapters_epoch_{epoch+1}.pt"
            save_lora_adapters(self.model, adapter_path)
            print(f"LoRA adapters saved to {adapter_path}")
        
        print("LoRA SFT training complete!")
        print("You can now share just the adapter files (few MB) instead of full model (GB)!")


if __name__ == "__main__":
    # - can use HIGHER learning rate (1e-4 vs 1e-5) because fewer params
    # - can use LARGER batch size (less memory per update)
    # - training is FASTER (fewer params to update)
    
    config = Config(
        model_name="llama3",
        version="lora_sft",
        max_sequence_length=512,
        embedding_size=512,
        num_attention_heads=8,
        num_layers=4,
        dropout_rate=0.1,
        learning_rate=1e-4,      # higher than full SFT (1e-5)!
        batch_size=8,             # can fit more in memory
        num_epochs=3,
        vocab_size=128000,
        tokenizer_type="tiktoken",
        num_kv_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=500000.0
    )
    
    # workflow:
    # 1. preprocess sft data (run once)
    # 2. train with LoRA (much cheaper than full fine-tuning)
    # 3. save adapters (tiny files you can share!)
    # 4. anyone can load your adapters into base model
    
    # example:
     trainer = LoRASFTTrainer(
         config,
         train_dataset_path="../sft/sft_data.json",
         pretrained_checkpoint="../pretraining/checkpoints/llama3_epoch_10.pt",
         lora_rank=16,    # typical for instruction following
         lora_alpha=32    # scaling factor
     )
     trainer.train()
    
    print("LoRA SFT training script ready!")
    print("Memory usage: ~10% of full fine-tuning")
    print("Training speed: ~2x faster than full fine-tuning")
    print("Adapter file size: ~20MB vs ~3GB for full model")


