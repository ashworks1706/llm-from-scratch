import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
import time 

print("CIFAR-10 IMAGE CLASSIFIER\n")

# 1. RESIDUAL BLOCK (reused from previous file)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 2. CIFAR-10 RESNET

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial conv: no stride=2, no maxpool (32x32 too small)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual layers: 32 → 64 → 128 channels
        self.layer1 = self._make_layer(32, 32, 3, stride=1)    # 32x32x32
        self.layer2 = self._make_layer(32, 64, 3, stride=2)    # 16x16x64
        self.layer3 = self._make_layer(64, 128, 3, stride=2)   # 8x8x128
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3. DATA LOADING

print("Loading CIFAR-10 dataset...")

# Training: augmentation + normalization
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # pad 4px border, crop random 32x32 (translation invariance)
    transforms.RandomHorizontalFlip(),     # 50% chance horizontal flip (not vertical - cars don't fly upside down)
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean/std per channel
])

# Test: just normalization (no augmentation on test set!)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]) 

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}\n")

# 4. TRAINING FUNCTIONS


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        
        if batch_idx % 50 == 0:
            print(f"  [{batch_idx:3d}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%")
    
    return total_loss / len(train_loader), correct / total

def validate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(test_loader), correct / total

# 5. TRAINING SETUP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

model = CIFAR10Net(num_classes=10).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
loss_fn = nn.CrossEntropyLoss()

# 6. TRAINING LOOP

num_epochs = 10  # set to 200 for full training
best_acc = 0.0

print("Starting training...\n")

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
    test_loss, test_acc = validate(model, test_loader, loss_fn, device)
    epoch_time = time.time() - start_time
    
    scheduler.step()
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
    print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")
    print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cifar10_model.pth')
        print(f"  ✓ New best accuracy: {best_acc*100:.2f}%")
    print()

print(f"\nTraining complete! Best test accuracy: {best_acc*100:.2f}%")


