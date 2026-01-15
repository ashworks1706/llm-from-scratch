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
    """
    Handles the complete pretraining pipeline for language models.
    
    Pretraining is the process of teaching a model to predict the next token in a sequence.
    The model learns language patterns, grammar, facts, and reasoning from raw text data.
    This is different from fine-tuning which adapts a pretrained model to specific tasks.
    
    The training process:
    1. Feed sequences of tokens to the model
    2. Model predicts next token at each position
    3. Compare predictions to actual next tokens (CrossEntropy loss)
    4. Backpropagate gradients to update weights
    5. Repeat millions of times until model learns language patterns
    """
    def __init__(self, config, train_dataset_path):
        self.config = config
        # Device selection: CUDA (GPU) for fast training, CPU as fallback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load preprocessed tokens from disk
        # These tokens were created by tokenizing raw text and saved for reuse
        tokens = torch.load(train_dataset_path)
        
        # Create dataset: converts token list into (input, target) pairs
        # Each sample: input=[tok1,tok2,tok3], target=[tok2,tok3,tok4] (shifted by 1)
        self.dataset = TextDataset(tokens, config.max_sequence_length)
        
        # DataLoader: batches samples together and shuffles for better training
        # Shuffling prevents model from memorizing order and helps generalization
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)
        
        # Initialize the Llama model with our config and move to GPU/CPU
        # .to(device) moves all model parameters to the specified device
        self.model = Llama(config).to(self.device)
        
        # AdamW optimizer: adaptive learning rate optimizer with weight decay
        # Adam adapts learning rate per parameter based on gradient history
        # Weight decay helps prevent overfitting by penalizing large weights
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # CrossEntropyLoss: combines softmax + negative log likelihood
        # Perfect for classification where we predict one token from vocabulary
        # It measures how different our predicted distribution is from the true label
        self.criterion = nn.CrossEntropyLoss()


    def train_step(self, inputs, targets):
        """
        Executes a single training step (one batch).
        
        This is the core of the training loop where learning actually happens:
        1. Forward pass: data flows through the model to get predictions
        2. Loss calculation: measure how wrong the predictions are
        3. Backward pass: compute gradients (how to adjust each weight)
        4. Weight update: use gradients to make model slightly better
        
        Args:
            inputs: Token IDs, shape (batch_size, seq_len)
            targets: Target token IDs (inputs shifted by 1), shape (batch_size, seq_len)
        
        Returns:
            Loss value as a float (for logging/monitoring)
        """
        # Move data to GPU/CPU - PyTorch operations need tensors on same device as model
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # 1. Zero gradients from previous step
        # PyTorch accumulates gradients by default, so we must clear old ones
        # Without this, gradients would keep adding up from every batch!
        self.optimizer.zero_grad()
        
        # 2. Forward pass - run data through the model
        # inputs: (batch, seq) → model processes → logits: (batch, seq, vocab)
        # Logits are raw scores for each token in vocabulary at each position
        # Higher logit = model thinks that token is more likely
        logits = self.model(inputs)
        
        # 3. Reshape for CrossEntropyLoss
        # CrossEntropyLoss expects: (batch*seq, vocab) and (batch*seq)
        # We flatten batch and sequence dimensions together
        # Why? Because each position is an independent prediction problem
        # Position 0 predicts token 1, position 1 predicts token 2, etc.
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)    # (batch*seq, vocab) - all predictions flattened
        targets = targets.view(-1)               # (batch*seq) - all labels flattened
        
        # 4. Calculate loss
        # CrossEntropyLoss: measures how different predicted distribution is from true label
        # It applies softmax to convert logits to probabilities, then computes negative log likelihood
        # Lower loss = better predictions. Goal is to minimize this value.
        # Example: if target token has 80% probability → loss is low
        #          if target token has 5% probability → loss is high
        loss = self.criterion(logits, targets)
        
        # 5. Backward pass - compute gradients
        # This calculates ∂loss/∂weight for every single parameter in the model
        # Tells us: "if we increase this weight by tiny amount, how much does loss change?"
        # These gradients are stored in each parameter's .grad attribute
        loss.backward()
        
        # 6. Update weights using gradients
        # Optimizer looks at all gradients and updates weights
        # Basic formula: weight = weight - learning_rate * gradient
        # Adam is smarter: it adapts learning rate per parameter based on gradient history
        self.optimizer.step()
        
        # 7. Return loss value as Python float (not tensor) for logging
        # .item() extracts the scalar value from a single-element tensor
        return loss.item()


    def train_epoch(self, epoch):
        """
        Trains for one complete pass through the dataset (one epoch).
        
        An epoch is one complete iteration through all training data.
        In practice, we need many epochs because:
        - One pass isn't enough to learn all patterns
        - Model needs to see examples multiple times to generalize
        - Each time it sees data, it understands patterns slightly better
        
        Args:
            epoch: Current epoch number (for logging)
        
        Returns:
            Average loss across all batches in this epoch
        """
        # Set model to training mode
        # This enables dropout, batch norm updates, etc. (behavior differs from eval mode)
        # Important because some layers behave differently during training vs inference
        self.model.train()
        
        total_loss = 0
        
        # Loop over all batches in the dataset
        # DataLoader automatically batches data and handles shuffling
        # Each iteration gives us (inputs, targets) for one batch
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            # Process one batch: forward, loss, backward, update
            loss = self.train_step(inputs, targets)
            total_loss += loss
            
            # Print progress every 10 batches
            # Helps us monitor training and catch issues early
            # If loss is NaN or exploding, we can stop and debug
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss:.4f}")
        
        # Calculate average loss for this epoch
        # This tells us overall how well model is doing on the training data
        # We expect this to decrease over epochs as model learns
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss


    def train(self):
        """
        Main training loop - orchestrates the entire training process.
        
        This runs multiple epochs of training and handles:
        - Progress monitoring across epochs
        - Checkpoint saving (so we can resume or use model later)
        - Overall training orchestration
        
        The model gets better with each epoch as it sees the data multiple times.
        We save checkpoints periodically so we don't lose progress if training stops.
        """
        print("Starting training...")
        
        # Create directory for saving model checkpoints
        # exist_ok=True means don't error if directory already exists
        os.makedirs("checkpoints", exist_ok=True)
        
        # Training loop: repeat for specified number of epochs
        # More epochs = more learning, but too many = overfitting (memorizing training data)
        for epoch in range(self.config.num_epochs):
            # Train for one complete pass through the data
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            # Checkpoints let us:
            # 1. Resume training if interrupted
            # 2. Use the model for inference later
            # 3. Compare different training stages
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/llama3_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),     # All model weights
                    'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state (momentum, etc.)
                    'loss': avg_loss,
                    'config': self.config  # Save config so we can recreate model architecture
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        print("Training complete!")


if __name__ == "__main__":
    """
    Example setup for pretraining a small Llama3 model.
    
    This config creates a "tiny" version for testing/learning:
    - Small dimensions (512) instead of full size (4096+)
    - Few layers (4) instead of many (32+)
    - Short sequences (128) instead of long (4096+)
    - Small batch size (4) for fitting in limited memory
    
    For real pretraining you'd use much larger values, but this is perfect for:
    - Understanding the training process
    - Testing code changes quickly
    - Running on consumer GPUs
    """
    # Create config for small model (for testing)
    config = Config(
        model_name="llama3",
        version="tiny",
        max_sequence_length=128,  # Short sequences for faster training
        embedding_size=512,       # Small embedding dimension (vs 4096 in full models)
        num_attention_heads=8,    # Number of attention heads
        num_layers=4,             # Few layers (vs 32+ in full models)
        dropout_rate=0.1,         # 10% dropout for regularization
        learning_rate=3e-4,       # Standard learning rate for Adam (0.0003)
        batch_size=4,             # Small batch size for limited GPU memory
        num_epochs=10,            # Number of complete passes through data
        vocab_size=128000,        # Llama3 vocabulary size (from tiktoken)
        tokenizer_type="tiktoken",
        num_kv_heads=4,           # GQA: 8 query heads share 4 key/value heads
        rms_norm_eps=1e-5,        # Epsilon for numerical stability in RMSNorm
        rope_theta=500000.0       # RoPE frequency base for positional encoding
    )
    
    # STEP 1: Preprocess data first (run this once)
    # This tokenizes raw text and saves tokens to disk
    # preprocessor = DataPreprocessor(config)
    # preprocessor.preprocess_text_file("path/to/your/text.txt", "data/tokens.pt")
    
    # STEP 2: Start training (after preprocessing is done)
    # This loads preprocessed tokens and trains the model
    # trainer = PreTrainer(config, "data/tokens.pt")
    # trainer.train()
    
    print("Setup complete. Uncomment the code above to preprocess and train.")

