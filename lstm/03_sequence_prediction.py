import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 


import yfinance as yf 

stock = yf.download("AAPL", start="2023-01-01", end="2025-01-01", progress=False)
prices = stock['Close'].values # closing prices 

# normalize prices
price_min = prices.min()
price_max = prices.max()

prices_normalized = (prices - price_min) / (price_max - price_min)

# create sequences
#

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 30
X, y = create_sequences(prices_normalized, seq_length)

print(X.shape, y.shape) # (num_samples, seq_length), (num_samples,)

# split into train and test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# convert to torch tensors
#
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (num_samples, seq_length, 1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)  # (num_samples, 1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
    def forward(self, x, h_prev, C_prev):
        # x: (batch, input_size), h_prev: (batch, hidden_size)
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_size + hidden_size)

        f_t = torch.sigmoid(self.W_f(combined))
        i_t = torch.sigmoid(self.W_i(combined))
        o_t = torch.sigmoid(self.W_o(combined))
        C_tilde = torch.tanh(self.W_c(combined))

        C_new = f_t * C_prev + i_t * C_tilde
        h_new = o_t * torch.tanh(C_new)

        return h_new, C_new


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        # batch_size, seq_length, _ = x.size() # error: too many values to unpack (expected 3)
        # fix :
        batch_size = x.size(0)
        seq_length = x.size(1)


        h = torch.zeros(batch_size, self.hidden_size)
        C = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            h, C = self.lstm_cell(x[:, t, :].squeeze(1), h, C)

        out = self.fc(h)
        return out



input_size = 1
hidden_size = 50
output_size = 1
model = LSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# evaluate on test set
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# plot training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.show()




# Applications of LSTMs are vast and varied. Some common applications include:
# 1. Time Series Forecasting: Predicting future values based on historical data, such as stock prices, weather data, or sales figures.
# 2. Natural Language Processing (NLP): Tasks like language modeling, text generation, machine translation, and sentiment analysis.
# 3. Speech Recognition: Converting spoken language into text by modeling the temporal dependencies in audio signals.
# 4. Anomaly Detection: Identifying unusual patterns in time series data, such as fraud detection in financial transactions or fault detection in industrial systems.
# 5. Video Analysis: Understanding and predicting sequences of frames in video data for applications like action recognition or video captioning.
# 6. Healthcare: Modeling patient data over time for predicting disease progression or treatment outcomes.
# 7. Robotics: Controlling robots by learning from sequences of sensor data and actions.
# 8. Music Generation: Creating new music by learning patterns from existing compositions.












