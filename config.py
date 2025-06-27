"""
Simple Configuration for COVID-19 Forecasting Project
"""

import torch
import numpy as np
import random

# ================================
# BASIC MODEL SETTINGS
# ================================

MODEL_PARAMS = {
    'rnn': {
        'input_dim': 4,          # 4 features: new_cases, tests_done, positivity_rate, population
        'hidden_dim': 8,        # Good balance of capacity vs overfitting
        'layer_dim': 2,          # 2 layers - enough complexity without overfitting
        'output_dim': 14,        # Predict next 14 days (sequence-to-sequence)
        'dropout': 0.3           # Standard dropout rate
    },
    
    'theta': {
        'period': 7              # Weekly seasonality
    }
}

# ================================
# TRAINING SETTINGS
# ================================

TRAINING_PARAMS = {
    'epochs': 18,                # Enough for convergence without overfitting
    'batch_size': 32,            # Good balance of stability vs memory usage
    'learning_rate': 1e-3,       # Standard Adam learning rate
    'weight_decay': 1e-6         # Light regularization
}

# ================================
# DATA SETTINGS
# ================================

DATA_PARAMS = {
    'window_size': 14,           # 2 weeks input
    'train_ratio': 0.7,          # 70% train
    'val_ratio': 0.15,           # 15% validation  
    'test_ratio': 0.15,          # 15% test
    'feature_columns': ['new_cases', 'tests_done', 'positivity_rate', 'population'],  # Multiple input features
    'target_column': 'new_cases',
    'countries': ['Germany', 'France', 'Italy', 'Spain', 'Belgium']
}

# ================================
# FILE PATHS
# ================================

PATHS = {
    'data': 'Data/Scaled_data.csv',  
    'results': {
        'plots': 'results/plots/',
        'models': 'results/models/',
        'reports': 'results/reports/'
    }
}

# ================================
# DEVICE & REPRODUCIBILITY
# ================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

print(f"Using device: {DEVICE}")