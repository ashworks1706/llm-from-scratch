import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
import os

# Add parent directory to path to import llama3 and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama3.model import Llama
from utils.config import Config
from dataset import SFTDataset, SFTDataPreprocessor
# in LoRA fientuning, instead of output = W @ input updating W entire matrix, we just add two small matrices A and B and keep W 
# frozen, like output = W @ input + (B @ A) input so total trainable params becomes 4096*r + r*4096 = ~65k vs 16M
# we only capture the small changes that happen in finetuning not retain everyting
# B @ A represents the delta change to the original weights 

class SFTTrainer:
    # SFT adapts a pretrained language model to follow instructions and give helpful responses.
    # The key difference from pretraining:
    # - Pretraining: model learns general language patterns from raw text
    # - SFT: model learns to map instructions → responses
    
    # Training process:
    # 1. Feed instruction + response as one sequence
    # 2. Only calculate loss on response tokens (mask instruction)
    # 3. Model learns: "given this instruction, generate this response"
    # 4. After SFT, model can follow new instructions it hasn't seen
    
    def __init__(self, config, train_dataset_path, pretrained_checkpoint=None):
        self.config = config
        
        # Device selection: CUDA (GPU) for fast training, CPU as fallback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create SFT dataset
        # Returns: (input_ids, target_ids, loss_mask) for each sample
        # loss_mask is key difference from pretraining!
        self.dataset = SFTDataset(train_dataset_path, config.max_sequence_length)
        
        # DataLoader: batches samples together and shuffles for better training
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        # Initialize model
        self.model = Llama(config).to(self.device)
        
        # Load pretrained weights if provided
        # This is the typical workflow: pretrain first, then SFT
        # Starting from pretrained weights is MUCH faster than training from scratch
        if pretrained_checkpoint:
            print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Pretrained weights loaded successfully!")
        else:
            print("No pretrained checkpoint provided. Training from scratch.")
        
        # AdamW optimizer: adaptive learning rate with weight decay
        # For SFT, we often use a LOWER learning rate than pretraining
        # Why? Because we're fine-tuning, not learning from scratch
        # Typical: pretraining LR = 3e-4, SFT LR = 1e-5 to 5e-5
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # CrossEntropyLoss with reduction='none'
        # reduction='none' gives us loss per position, so we can apply mask
        # This is THE key difference from pretraining!
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def train_step(self, inputs, targets, loss_mask):
        # Executes a single training step with masked loss.
        
        # The ONLY difference from pretraining train_step:
        # - We receive loss_mask as third argument
        # - We apply mask to loss before averaging
        # - This ensures we only train on response tokens, not instruction tokens
        
        # loss_mask: Mask for loss calculation, shape (batch_size, seq_len)
        # 0 = ignore (instruction tokens)
        # = calculate loss (response tokens)
        
        # Move data to GPU/CPU
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        loss_mask = loss_mask.to(self.device)
        
        # 1. Zero gradients from previous step
        self.optimizer.zero_grad()
        
        # 2. Forward pass - run data through model
        # inputs: (batch, seq) → logits: (batch, seq, vocab)
        logits = self.model(inputs)
        
        # 3. Reshape for CrossEntropyLoss
        # Need to flatten batch and sequence dimensions
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)  # (batch*seq, vocab)
        targets = targets.view(-1)             # (batch*seq)
        loss_mask = loss_mask.view(-1)         # (batch*seq)
        
        # 4. Calculate loss PER POSITION (not averaged yet!)
        # reduction='none' means we get loss for each position separately
        # Shape: (batch*seq) - one loss value per token position
        loss_per_position = self.criterion(logits, targets)
        
        # 5. Apply mask - THIS IS THE SFT-SPECIFIC STEP!
        # Multiply loss by mask to zero out instruction tokens
        # Instruction tokens have mask=0, so their loss becomes 0
        # Response tokens have mask=1, so their loss stays the same
        # Example: loss=[0.5, 0.3, 0.4, 0.6] * mask=[0, 0, 1, 1] = [0.0, 0.0, 0.4, 0.6]
        masked_loss = loss_per_position * loss_mask
        
        # 6. Average over response tokens ONLY
        # We sum all (masked) losses and divide by number of response tokens
        # mask.sum() tells us how many response tokens (mask=1 positions)
        # This gives accurate loss that reflects response quality
        # If we divided by all positions, loss would be artificially low!
        loss = masked_loss.sum() / loss_mask.sum()
        
        # 7. Backward pass - compute gradients
        # Gradients only flow through response tokens because instruction losses are zero
        loss.backward()
        
        # 8. Update weights
        self.optimizer.step()
        
        # 9. Return loss value as Python float
        return loss.item()
    
    def train_epoch(self, epoch):
        # Set model to training mode
        self.model.train()
        
        total_loss = 0
        
        # Loop over all batches
        # Note: SFTDataset returns 3 items (inputs, targets, loss_mask)
        for batch_idx, (inputs, targets, loss_mask) in enumerate(self.dataloader):
            # Process one batch with masked loss
            loss = self.train_step(inputs, targets, loss_mask)
            total_loss += loss
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss:.4f}")
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss
    
    def train(self):
        # - Same structure as pretraining, but typically:
        # - Fewer epochs (3-5 for SFT vs 10+ for pretraining)
        # - Lower learning rate (we're fine-tuning, not learning from scratch)
        # - Smaller dataset (thousands vs billions of tokens)
        
        print("Starting SFT training...")
        
        # Create directory for saving checkpoints
        os.makedirs("sft_checkpoints", exist_ok=True)
        
        # Training loop: fewer epochs than pretraining
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every epoch (SFT is fast, so we save more frequently)
            checkpoint_path = f"sft_checkpoints/llama3_sft_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
                'config': self.config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        print("SFT training complete!")
        print("Model is now instruction-tuned and ready to follow commands!")


if __name__ == "__main__":
    #  Key differences from pretraining config:
    # LOWER learning rate (1e-5 vs 3e-4) - fine-tuning needs gentler updates
    # FEWER epochs (3-5 vs 10+) - SFT dataset is smaller and converges faster
    # Can use LONGER sequences if needed (instructions can be long)
    
    Typical workflow:
    # 1. Pretrain model on large text corpus (expensive, takes days/weeks)
    # 2. SFT on instruction data (cheap, takes hours)
    # 3. Optional: RLHF for alignment (we'll do this next!)
    
    # Create config for SFT
    # Note: Lower learning rate than pretraining!
    config = Config(
        model_name="llama3",
        version="sft",
        max_sequence_length=512,  # Can be longer for complex instructions
        embedding_size=512,
        num_attention_heads=8,
        num_layers=4,
        dropout_rate=0.1,
        learning_rate=1e-5,       # Much lower than pretraining (3e-4)!
        batch_size=4,
        num_epochs=3,              # Fewer epochs than pretraining
        vocab_size=128000,
        tokenizer_type="tiktoken",
        num_kv_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=500000.0
    )
    
    # STEP 1: Preprocess your SFT data (run once)
    # Converts raw data to standard format
    # Uncomment to preprocess:
    # preprocessor = SFTDataPreprocessor()
    # preprocessor.preprocess_conversations("raw_sft_data.json", "sft_data.json")
    
    # STEP 2: Start SFT training
    # Load pretrained checkpoint and fine-tune on instructions
    # Uncomment to train:
    # trainer = SFTTrainer(
    #     config,
    #     train_dataset_path="sft_data.json",
    #     pretrained_checkpoint="checkpoints/llama3_epoch_10.pt"  # Load pretrained model
    # )
    # trainer.train()
    
    print("Setup complete. Uncomment the code above to preprocess and train SFT.")
