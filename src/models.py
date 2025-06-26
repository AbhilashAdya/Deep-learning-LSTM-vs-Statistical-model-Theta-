"""
Models for COVID-19 Forecasting: RNN and Theta Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from config import MODEL_PARAMS, TRAINING_PARAMS, DEVICE

# ================================
# RNN MODEL DEFINITION
# ================================

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(RNNModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # RNN layers
        self.rnn = nn.LSTM(  # Using LSTM instead of basic RNN
            input_dim, 
            hidden_dim, 
            layer_dim, 
            batch_first=True, 
            dropout=dropout if layer_dim > 1 else 0  # Only dropout if multiple layers
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Initialize hidden state
        device = x.device
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        
        # Forward propagation through LSTM
        lstm_out, _ = self.rnn(x, (h0, c0))
        
        # Apply fully connected layer to last time step
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # We want the last time step: lstm_out[:, -1, :]
        output = self.fc(lstm_out[:, -1, :])  # Shape: (batch_size, output_dim)
        
        return output

# ================================
# RNN TRAINER CLASS
# ================================

class RNNTrainer:
    def __init__(self, input_dim=None, hidden_dim=None, layer_dim=None, 
                 output_dim=None, dropout=None):
        """Initialize RNN trainer with model parameters"""
        
        # Use config values if not provided
        if input_dim is None:
            input_dim = MODEL_PARAMS['rnn']['input_dim']
        if hidden_dim is None:
            hidden_dim = MODEL_PARAMS['rnn']['hidden_dim']
        if layer_dim is None:
            layer_dim = MODEL_PARAMS['rnn']['layer_dim']
        if output_dim is None:
            output_dim = MODEL_PARAMS['rnn']['output_dim']
        if dropout is None:
            dropout = MODEL_PARAMS['rnn']['dropout']
        
        # Create model
        self.model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
        self.model = self.model.to(DEVICE)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=TRAINING_PARAMS['learning_rate']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        print(f"RNN Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Reshape inputs if needed
            # Expected input shape: (batch_size, seq_len, input_dim)
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(-1)  # Add feature dimension
            
            # Forward pass
            predictions = self.model(inputs)
            
            # For sequence-to-sequence, we need to match target shape
            # targets shape: (batch_size, seq_len)
            # predictions shape: (batch_size, output_dim)
            
            # If output_dim > 1, we're predicting a sequence
            if predictions.shape[1] > 1:
                loss = self.criterion(predictions, targets)
            else:
                # If output_dim = 1, we're predicting single values
                # Take mean of target sequence for single prediction
                targets_mean = targets.mean(dim=1, keepdim=True)
                loss = self.criterion(predictions, targets_mean)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Reshape inputs if needed
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(-1)
                
                predictions = self.model(inputs)
                
                # Handle output shape matching
                if predictions.shape[1] > 1:
                    loss = self.criterion(predictions, targets)
                else:
                    targets_mean = targets.mean(dim=1, keepdim=True)
                    loss = self.criterion(predictions, targets_mean)
                
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def train(self, train_loader, val_loader, epochs=None):
        """Complete training loop"""
        
        if epochs is None:
            epochs = TRAINING_PARAMS['epochs']
        
        print(f"Starting RNN training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print("RNN training completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1]
        }
    
    def predict(self, test_loader):
        """Make predictions on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                
                predictions = self.model(inputs)
                predictions = predictions.squeeze(-1)
                
                # Move to CPU and convert to numpy
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RNN Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ================================
# THETA MODEL TRAINER CLASS
# ================================

class ThetaTrainer:
    def __init__(self, period=None):
        """Initialize Theta model trainer"""
        
        if period is None:
            period = MODEL_PARAMS['theta']['period']
        
        self.period = period
        self.predictions = []
        self.targets = []
        self.losses = []
        
        print(f"Theta Model created with period={period}")
    
    def train_and_predict(self, data_loader, split_name=""):
        """Train Theta model and make predictions"""
        
        print(f"Training Theta model on {split_name} data...")
        
        all_predictions = []
        all_targets = []
        batch_losses = []
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch_predictions = []
            
            # Process each sequence in the batch
            for seq_idx in range(inputs.shape[0]):
                # Get input sequence (convert to 1D if needed)
                input_seq = inputs[seq_idx]
                if len(input_seq.shape) > 1:
                    input_seq = input_seq[:, 0]  # Take first feature if multi-dimensional
                
                target_seq = targets[seq_idx]
                
                try:
                    # Convert to numpy and handle numerical issues
                    input_data = input_seq.numpy()
                    
                    # Check for problematic data
                    if np.all(input_data == 0) or np.isnan(input_data).any() or np.isinf(input_data).any():
                        # Use simple persistence model as fallback
                        forecast = np.full(len(target_seq), input_data[-1] if len(input_data) > 0 else 0.0)
                    else:
                        # Add small noise to avoid constant series issues
                        if np.std(input_data) < 1e-6:
                            input_data = input_data + np.random.normal(0, 1e-6, len(input_data))
                        
                        # Fit Theta model with error handling
                        theta_model = ThetaModel(input_data, period=self.period)
                        fitted_model = theta_model.fit(disp=False)  # Suppress optimization messages
                        
                        # Forecast next sequence
                        forecast = fitted_model.forecast(len(target_seq))
                        
                        # Handle any remaining NaN/inf values
                        if np.isnan(forecast).any() or np.isinf(forecast).any():
                            forecast = np.full(len(target_seq), input_data[-1])
                    
                    batch_predictions.append(forecast)
                    
                except Exception as e:
                    # If Theta model fails completely, use persistence
                    last_value = input_seq[-1].item() if len(input_seq) > 0 else 0.0
                    forecast = np.full(len(target_seq), last_value)
                    batch_predictions.append(forecast)
            
            # Convert to numpy array
            batch_predictions = np.array(batch_predictions)
            batch_targets = targets.numpy()
            
            # Calculate batch loss (handle any remaining numerical issues)
            try:
                batch_loss = mean_squared_error(batch_targets, batch_predictions)
                if np.isnan(batch_loss) or np.isinf(batch_loss):
                    batch_loss = 1.0  # Default loss if calculation fails
            except:
                batch_loss = 1.0
            
            batch_losses.append(batch_loss)
            
            all_predictions.append(batch_predictions)
            all_targets.append(batch_targets)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1} - Loss: {batch_loss:.4f}")
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mean_loss = np.mean(batch_losses)
        print(f"Theta {split_name} completed - Mean Loss: {mean_loss:.4f}")
        
        return predictions, targets, mean_loss
    


# ================================
# MODEL FACTORY FUNCTIONS
# ================================

def create_rnn_trainer(**kwargs):
    """Factory function to create RNN trainer"""
    return RNNTrainer(**kwargs)

def create_theta_trainer(**kwargs):
    """Factory function to create Theta trainer"""
    return ThetaTrainer(**kwargs)