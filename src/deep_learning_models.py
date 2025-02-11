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
    def __init__(self, input_shape, filters=[64, 32], kernel_size=3):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        # First Conv1D layer
        self.conv_layers.append(
            nn.Conv1d(input_shape[1], filters[0], kernel_size)
        )
        self.conv_layers.append(nn.BatchNorm1d(filters[0]))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool1d(2))
        
        # Additional Conv1D layers
        for i in range(len(filters)-1):
            self.conv_layers.append(
                nn.Conv1d(filters[i], filters[i+1], kernel_size)
            )
            self.conv_layers.append(nn.BatchNorm1d(filters[i+1]))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(2))
        
        # Calculate the size of flattened features
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(input_shape), 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        for layer in self.conv_layers:
            x = layer(x)
        return x.numel()
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        return self.fc(x)

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
    def __init__(self, input_shape, units=[64, 32]):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        
        # LSTM layers
        self.lstm = nn.LSTM(
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
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output
        return self.fc(x)

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