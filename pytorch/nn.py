import torch
import torch.nn as nn 
import torch.nn.functional as F 


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b1 = nn.Parameter(torch.randn(hidden_size))
        self.w2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b2 = nn.Parameter(torch.randn(output_size))

    def forward(self,x):

        h = x @ self.w1 + self.b1 
        h= torch.relu(h)

        output = h @ self.w2 + self.b2 

        return output 


twolayernt = TwoLayerNet(input_size=4, hidden_size=8, output_size=2)
x = torch.randn(3,4)

output = twolayernt(x)

print(twolayernt)
print(x)
print(output)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias= None

    def forward(self, x):
        output = x @ self.weight.T 
        if self.bias is not None:
            output= output +self.bias 

        return output 

linear = Linear(in_features=4, out_features=2) 
x= torch.randn(3,4)
out = linear(x)
print(linear)
print(x)
print(out)




class NeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden1=16, hidden2=8, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x= F.relu(x)
        x= self.fc3(x)

        return x 


model = NeuralNetwork()

print(model)


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct= 0
    total = 0 

    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        preds = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred_c = preds.argmax(dim=1)
        correct+= (pred_c == targets).sum().item()
        total+= targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct/ total
    return avg_loss, accuracy
