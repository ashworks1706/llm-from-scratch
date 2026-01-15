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
from dataset import TextDataset, DataPreprocessor

class PreTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda")
        self.model = config.model
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        # config = Config(
                model_name="llama3",
                version="tiny",
                max_sequence_length=128,  # Small for testing
                embedding_size=512,       # Small for testing
                num_attention_heads=8,
                num_layers=4,             # Small for testing
                dropout_rate=0.1,
                learning_rate=3e-4,
                batch_size=4,             # Small for testing
                num_epochs=10,
                vocab_size=128000,        # Llama3 vocab size
                tokenizer_type="tiktoken",
                num_kv_heads=4,           # GQA: 8 query heads, 4 kv heads
                rms_norm_eps=1e-5,
                rope_theta=500000.0
            )


    def train_step(self, model, inputs, targets, optimizer, device):
        
        # Move data to device
        inputs.to(self.device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass - get logits

        
        # TODO: 3. Reshape for CrossEntropyLoss
        # logits: (batch, seq, vocab) -> (batch*seq, vocab)
        # targets: (batch, seq) -> (batch*seq)
        
        # TODO: 4. Calculate loss
        
        # TODO: 5. Backward pass - compute gradients
        
        # TODO: 6. Update weights
        
        # TODO: 7. Return loss value
        pass


    def train_epoch(self, model, dataloader, optimizer, device, epoch):
        model.train()  # Set model to training mode
        total_loss = 0
        
        # TODO: Loop over batches
        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        #     loss = train_step(...)
        #     total_loss += loss
        #     
        #     # Print progress every N batches
        #     if batch_idx % 10 == 0:
        #         print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        # TODO: Calculate and return average loss
        pass


    def train(self, config, train_dataset_path):
    
        # TODO: 1. Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # TODO: 2. Load preprocessed tokens
        # tokens = torch.load(train_dataset_path)
        
        # TODO: 3. Create dataset and dataloader
        # dataset = TextDataset(tokens, config.max_sequence_length)
        # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        # TODO: 4. Initialize model
        # model = Llama(config).to(device)
        
        # TODO: 5. Initialize optimizer
        # optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        
        # TODO: 6. Training loop over epochs
        # for epoch in range(config.num_epochs):
        #     avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        #     print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        #     
        #     # TODO: Save checkpoint every N epochs
        #     if (epoch + 1) % 5 == 0:
        #         checkpoint_path = f"checkpoints/llama3_epoch_{epoch+1}.pt"
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': avg_loss,
        #         }, checkpoint_path)
        #         print(f"Checkpoint saved to {checkpoint_path}")
        
        print("Training complete!")
        pass


if __name__ == "__main__":
    # TODO: Create config for small model (for testing)
    config = Config(
        model_name="llama3",
        version="tiny",
        max_sequence_length=128,  # Small for testing
        embedding_size=512,       # Small for testing
        num_attention_heads=8,
        num_layers=4,             # Small for testing
        dropout_rate=0.1,
        learning_rate=3e-4,
        batch_size=4,             # Small for testing
        num_epochs=10,
        vocab_size=128000,        # Llama3 vocab size
        tokenizer_type="tiktoken",
        num_kv_heads=4,           # GQA: 8 query heads, 4 kv heads
        rms_norm_eps=1e-5,
        rope_theta=500000.0
    )
    
    # TODO: Preprocess data first (if not already done)
    # preprocessor = DataPreprocessor(config)
    # preprocessor.preprocess_text_file("path/to/your/text.txt", "data/tokens.pt")
    
    # TODO: Start training
    # train(config, "data/tokens.pt")
