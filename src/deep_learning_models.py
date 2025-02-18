import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPModel(nn.Module):
    def __init__(self, input_dim, layers=[64, 32], dropout_rate=0.3):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, layers[0]))
        self.layers.append(nn.BatchNorm1d(layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.BatchNorm1d(layers[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(layers[-1], 1))
        self.layers.append(nn.Sigmoid())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape, filters=64, kernel_size=3):
        super(CNNModel, self).__init__()
        
        # Unpack input shape (timesteps, features)
        n_channels, n_features = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=n_channels,  # number of timesteps as channels
                     out_channels=filters, 
                     kernel_size=kernel_size,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=filters,
                     out_channels=filters*2,
                     kernel_size=kernel_size,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1)
        )
        
        # Calculate the size of flattened features
        self.flatten = nn.Flatten()
        
        # Use _get_conv_output to calculate the input size for the fully connected layer
        conv_output_size = self._get_conv_output((1, n_channels, n_features))
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _get_conv_output(self, shape):
        x = torch.zeros(shape)
        x = self.conv_layers(x)
        return x.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_shape, units=[64, 32]):
        super(RNNModel, self).__init__()
        self.rnn_layers = nn.ModuleList()
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_shape[1],
            hidden_size=units[0],
            num_layers=len(units),
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(units[0], 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last output
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        
        # Unpack input shape (timesteps, features)
        self.seq_length, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        return out

def train_evaluate_dl_model(
    model, model_name, train_loader, test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epochs=100, patience=10
):
    """
    Train and evaluate a deep learning model
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    best_auc = 0
    patience_counter = 0
    
    with mlflow.start_run(run_name=f"dl_{model_name}"):
        train_losses = []
        val_aucs = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            # Validation
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(y_batch.numpy())
            
            val_auc = roc_auc_score(all_targets, all_preds)
            val_aucs.append(val_auc)
            
            print(f'Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f} - Val AUC: {val_auc:.4f}')
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), f'../models/best_{model_name}.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(val_aucs, label='Validation AUC')
        plt.title(f'{model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(f'../models/{model_name}_training_history.png')
        mlflow.log_artifact(f'../models/{model_name}_training_history.png')
        plt.close()
        
        return model, best_auc

def prepare_sequence_data(X, timesteps):
    """
    Prepare sequential data for RNN/LSTM/CNN models
    """
    n_samples = X.shape[0] - timesteps + 1
    n_features = X.shape[1]
    sequences = np.zeros((n_samples, timesteps, n_features))
    
    for i in range(n_samples):
        sequences[i] = X[i:i+timesteps]
    
    return sequences 