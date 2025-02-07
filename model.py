import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Output last time step


# Load Stock Data
def load_stock_data(ticker, start="2015-01-01"):
    stock_data = yf.download(ticker, start=start)
    data = stock_data[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler


# Prepare Data for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Train LSTM Model
def train_model(ticker):
    data, scaler = load_stock_data(ticker)
    X, y = create_sequences(data)

    X_train, y_train = torch.tensor(X[:-100], dtype=torch.float32), torch.tensor(y[:-100], dtype=torch.float32)
    X_test, y_test = torch.tensor(X[-100:], dtype=torch.float32), torch.tensor(y[-100:], dtype=torch.float32)

    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    return model, scaler


# Predict Future Prices
def predict_stock(model, scaler, ticker):
    data, _ = load_stock_data(ticker)
    X, _ = create_sequences(data)

    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(X[-1:], dtype=torch.float32))

    return scaler.inverse_transform(prediction.numpy())[0][0]
