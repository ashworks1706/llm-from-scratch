# complete training loop example

# Load data, Define model, Compute loss, backpropagate, Update weight, Track progress, validate
# L D C B U T V 


# mainly 
# forward pass : feed data -> compute predictions -> compute loss 
# backward pass : compute gradients of loss wrt all parameters 
# optimizer : update parameters using gradients 
# repeat 

# epoch -> one complete pass through dataset 
#   inner loop -> batches 
#       process one mini batch data:input features, targets: ground truth labels 

# why batches?
# can't load entire dataset in memory, 
# Option 1: One sample at a time (batch_size=1)
# ✓ Memory efficient
# ✗ Very slow, noisy gradients

# Option 2: Entire dataset (batch_size=N)
# ✗ Out of memory
# ✗ Slow, stuck in local minima

# Option 3: Mini-batches (batch_size=32-256) ✓
# ✓ Fits in memory
# ✓ Fast (parallel computation)
# ✓ Good gradient estimates
# ✓ Noise helps generalization


# data loader class from torch we use it because it does automatic batching which is nice 
# we shuffle data, why shuffle?
# Without shuffle:
# Batch 1: All class 0 samples
# Batch 2: All class 1 samples
# → Model sees patterns in order, learns slowly

# With shuffle:
# Batch 1: Mixed classes
# Batch 2: Mixed classes
# → Model gets diverse examples, learns faster

# Training mode
# model.train()
# - Dropout active (random neuron dropping)
# - BatchNorm uses batch statistics
# - Gradients computed

# Evaluation mode
# model.eval()
# - Dropout off (use all neurons)
# - BatchNorm uses running statistics
# - No gradients (faster, less memory)


# gradient clipping is also needed to prevent exploding gradients, by scaling down large gradeints while skeeping direction 


# why do we do validaiton seperately?
# because training data: model learns from it, validation data: check if model generalizes well on unseen data 
# if training loss is less but validation loss is high: that means the model is overfitting and not learning,
# u can try regularization, more data or simpler model 
# if the loss is NaN, it could because of leanring rate is too high or numerical instability
# the soluton to this would be to lower learning rate, gradient clipping and data validation 

# if the loss is not decreasing, maybe its because LR is too low, wrong Loss function, or data not noramlized
# usualyl the soluton to this is increasing LR, looking for data shapes, and normalizing them 
#
# if the model is overfitting, that means training loss is less but validation loss is hihg 
# solutions could be adding dropout, weight decay (l2 regularization), getting more data, or early stopping (stop when validation loss stops improving)





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


print("1. DATASET PREPARATION")
#
class SyntheticDataset(Dataset):
    # dataset for binary classification
    # Task: Given 2D point (x, y), classify if it's in circle or not
    
    # Circle equation: x² + y² < radius²
    # Inside circle → class 0
    # Outside circle → class 1
    def __init__(self, num_samples=1000, radius=1.0):
        # Generate random 2D points in range [-2, 2]
        self.data = torch.randn(num_samples, 2) * 2
        
        # Label: inside circle (0) or outside (1)
        distances = torch.sum(self.data ** 2, dim=1)  # x² + y²
        self.labels = (distances > radius ** 2).long()  # 1 if outside, 0 if inside
        
        # Store for info
        self.num_samples = num_samples
        self.radius = radius
        
    def __len__(self):
        # Required: return dataset size
        return self.num_samples
    
    def __getitem__(self, idx):
        # Required: return single sample
        # DataLoader calls this to build batches
        return self.data[idx], self.labels[idx]

# Create datasets
train_dataset = SyntheticDataset(num_samples=1000, radius=1.0)
val_dataset = SyntheticDataset(num_samples=200, radius=1.0)

print(f" Created training dataset: {len(train_dataset)} samples")
print(f" Created validation dataset: {len(val_dataset)} samples")

# Check class balance
train_class_0 = (train_dataset.labels == 0).sum().item()
train_class_1 = (train_dataset.labels == 1).sum().item()
print(f"\n  Training class distribution:")
print(f"    Class 0 (inside):  {train_class_0} samples ({train_class_0/len(train_dataset)*100:.1f}%)")
print(f"    Class 1 (outside): {train_class_1} samples ({train_class_1/len(train_dataset)*100:.1f}%)")

# DataLoader handles:
# - Automatic batching
# - Shuffling
# - Parallel data loading (num_workers)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,      # Process 32 samples at once
    shuffle=True,       # Randomize order each epoch (important!)
    num_workers=0       # use 2-4 for real training
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,      # Don't shuffle validation (not necessary)
    num_workers=0
)

print(f"    Training batches per epoch: {len(train_loader)}")
print(f"    Validation batches: {len(val_loader)}")

# 2. DEFINE MODEL

class SimpleClassifier(nn.Module):
    # Architecture: Input(2) → Hidden(16) → Hidden(8) → Output(2)
    def __init__(self, input_size=2, hidden1=16, hidden2=8, num_classes=2):
        super().__init__()
        
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_size, hidden1)
        
        # Layer 2: Hidden → Hidden
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        # Layer 3: Hidden → Output
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        # Store architecture for logging
        self.architecture = f"{input_size}→{hidden1}→{hidden2}→{num_classes}"
        
    def forward(self, x):
        # x: (batch_size, 2)
        
        # Layer 1 + ReLU activation
        x = self.fc1(x)      # (batch, 16)
        x = F.relu(x)        # Non-linearity
        
        # Layer 2 + ReLU activation
        x = self.fc2(x)      # (batch, 8)
        x = F.relu(x)
        
        # Layer 3 (output logits)
        x = self.fc3(x)      # (batch, 2)
        # Note: No softmax here! CrossEntropyLoss applies it internally
        
        return x

# Create model
model = SimpleClassifier()

print(f"✓ Model architecture: {model.architecture}")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print(f"\n  Parameter  :")
for name, param in model.named_parameters():
    print(f"    {name:20s}: {param.shape} = {param.numel():,} params")

# 3. SETUP TRAINING COMPONENTS

# Loss function
# CrossEntropyLoss for classification
# Combines LogSoftmax + NLLLoss
# Expects: logits (raw scores) and class indices
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,              # Learning rate
    weight_decay=0.01     # L2 regularization
)

# Reduces learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Minimize validation loss
    factor=0.5,          # Multiply lr by 0.5
    patience=5,          # Wait 5 epochs before reducing
    verbose=True         # Print when lr changes
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. TRAINING LOOP

def train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
   
    # Set model to training mode
    # - Enables dropout (if we had it)
    # - BatchNorm uses batch statistics (if we had it)
    model.train()
    
    # Track metrics
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate through batches
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # STEP 1: Forward pass
        predictions = model(data)  # (batch_size, num_classes)
        loss = loss_fn(predictions, targets)
        
        # STEP 2: Backward pass
        optimizer.zero_grad()  # Clear old gradients (CRITICAL!)
        loss.backward()        # Compute gradients
        
        # Optional: Gradient clipping (prevent exploding gradients)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # STEP 3: Optimizer step
        optimizer.step()  # Update all parameters
        
        # STEP 4: Logging
        total_loss += loss.item()
        
        # Calculate accuracy
        pred_classes = predictions.argmax(dim=1)  # Get predicted class
        correct += (pred_classes == targets).sum().item()
        total += targets.size(0)
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx:3d}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100*correct/total:.2f}%")
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, loss_fn, device):
    # Set model to evaluation mode
    # - Disables dropout
    # - BatchNorm uses running statistics
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Don't compute gradients 
    with torch.no_grad():
        for data, targets in val_loader:
            # Move to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass only (no backward!)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            # Track metrics
            total_loss += loss.item()
            pred_classes = predictions.argmax(dim=1)
            correct += (pred_classes == targets).sum().item()
            total += targets.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Training configuration
num_epochs = 20
best_val_loss = float('inf')

print(f"\nTraining for {num_epochs} epochs...\n")

# Store history for plotting
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Main training loop
for epoch in range(1, num_epochs + 1):
    print(f"EPOCH {epoch}/{num_epochs}")
    print(f"{'='*70}")
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, loss_fn, optimizer, device, epoch
    )
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print epoch summary
    print(f"\n  Epoch {epoch} Summary:")
    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    
    # Check if best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save best model (in real training, you'd save to disk)
        print(f"    ✓ New best validation loss! Saving model...")
        best_model_state = model.state_dict().copy()
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Early stopping check (optional)
    # If validation loss hasn't improved in 10 epochs, stop
    if epoch > 10:
        recent_val_losses = history['val_loss'][-10:]
        if all(loss >= best_val_loss for loss in recent_val_losses):
            print(f"\n  Early stopping: No improvement in 10 epochs")
            break

# 5. FINAL EVALUATION

print("\n" + "="*70)
print("5. FINAL RESULTS")

# Load best model
model.load_state_dict(best_model_state)

# Final validation
final_val_loss, final_val_acc = validate(model, val_loader, loss_fn, device)

print(f"\n  Best model performance:")
print(f"    Validation Loss: {final_val_loss:.4f}")
print(f"    Validation Accuracy: {final_val_acc*100:.2f}%")

# Show training progress
print(f"\n  Training progress:")
print(f"    Train Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
print(f"    Train Acc:  {history['train_acc'][0]*100:.2f}% → {history['train_acc'][-1]*100:.2f}%")
print(f"    Val Loss:   {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
print(f"    Val Acc:    {history['val_acc'][0]*100:.2f}% → {history['val_acc'][-1]*100:.2f}%")

# 6. TEST INFERENCE
# ============================================================================

print("\n" + "="*70)
print("6. INFERENCE EXAMPLE")

test_points = torch.tensor([
    [0.0, 0.0],    # Center (inside)
    [0.5, 0.5],    # Inside
    [2.0, 2.0],    # Outside
    [-1.5, -1.5],  # Outside
])

model.eval()
with torch.no_grad():
    test_points_device = test_points.to(device)
    logits = model(test_points_device)
    probabilities = F.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)

print("\n  Testing specific points:")

for i, point in enumerate(test_points):
    x, y = point
    true_radius = (x**2 + y**2).sqrt().item()
    pred_class = predictions[i].item()
    confidence = probabilities[i][pred_class].item()
    pred_label = "inside" if pred_class == 0 else "outside"
    
    print(f"  ({x:5.2f}, {y:5.2f})  r={true_radius:<6.2f}     {pred_label:<10s}  {confidence*100:5.2f}%")




















































































































































































































