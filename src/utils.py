import torch
from torch.utils.data import Dataset, DataLoader

def windowing(data, features, target, tw):

    """
    Create sequences with multiple input features but single target
    """

    inout_seq = []
    L = len(data)


    for i in range(L-2*tw+1):
        
        input_features = data[features].iloc[i:i+tw].values
        target_values = data[target].iloc[i+tw:i+2*tw].values
        
        inout_seq.append((input_features, target_values))
        
    return inout_seq


def split_data(data, train_ratio=0.7, validation_ratio=0.15):
    
    total_data =len(data)

    train_data = data[:int(total_data * train_ratio)]
    validation_data = data[int(total_data * train_ratio): int(total_data * (train_ratio + validation_ratio))]
    test_data = data[int(total_data * (train_ratio + validation_ratio)):]

    return train_data, validation_data, test_data



class TimeSeriesDataset(Dataset):

    """
        This class takes your windowed sequences and makes them PyTorch-ready.
        
        What it does:
        - Takes list of (input, target) tuples
        - Converts them to PyTorch tensors when needed
        - Allows PyTorch to work with your data
    """

    def __init__(self, sequences):
        """
        Store the sequences.
        sequences = [(input1, target1), (input2, target2), ...]
        """
        self.sequences = sequences
    
    def __len__(self):
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
       
        """Return the input and target sequence at index idx."""
        
        input_seq, target_seq = self.sequences[idx]
        input_tensor = torch.FloatTensor(input_seq)
        target_tensor = torch.FloatTensor(target_seq)
        
        return input_tensor, target_tensor
    
def create_dataloader(train_sequences, val_sequences, test_sequences, batch_size=32, shuffle=True):
    """
        Create DataLoaders for train, validation, and test sets.
        
        What it does:
        - Takes your 3 sets of sequences
        - Creates 3 separate DataLoaders in one function
        - Sets shuffle=True for training, False for val/test
        
        Parameters:
        - train_sequences, val_sequences, test_sequences: From windowing function
        - batch_size: How many sequences per batch
        """
    train_dataset = TimeSeriesDataset(train_sequences)
    val_dataset = TimeSeriesDataset(val_sequences)
    test_dataset = TimeSeriesDataset(test_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    "Note: Now, when you itereate through the DataLoader, __git__items would return a batch of input and target sequences as tensors."

    return train_loader, val_loader, test_loader


def prepare_data(raw_data, features=['new_cases'], target='new_cases', window_size=14, batch_size=32):
    """
    Complete pipeline: raw data → ready DataLoaders
    
    What it does:
    1. Split your raw time series data
    2. Create windowed sequences for each split
    3. Create DataLoaders for each split
    4. Return everything ready for model training
    
    Parameters:
    - raw_data: Your DataFrame with time series data
    - features: List of feature column names to use as input
    - target: Target column name to predict
    - window_size: How many time steps in each sequence
    - batch_size: How many sequences per batch
    """
    # Step 1: Split the data
    train_data, val_data, test_data = split_data(raw_data, train_ratio=0.7, validation_ratio=0.15)

    # Step 2: Create windowed sequences
    training_window = windowing(train_data, features, target, window_size)
    validation_window = windowing(val_data, features, target, window_size)
    test_window = windowing(test_data, features, target, window_size)
    
    # Step 3: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloader(
        training_window, validation_window, test_window, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader

