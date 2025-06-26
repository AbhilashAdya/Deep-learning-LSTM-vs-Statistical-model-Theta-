import pandas as pd
from src.models import create_rnn_trainer, create_theta_trainer
from src.utils import prepare_data
from src.evaluation import ModelEvaluator
from config import DATA_PARAMS, TRAINING_PARAMS, PATHS

def load_data():
    """Load processed data from notebook"""
    print(" Loading processed data...")
    
    try:
        data = pd.read_csv(PATHS['data'])
        data['datetime'] = pd.to_datetime(data['datetime'])
        print(f" Data loaded: {data.shape}")
        print(f"Countries: {list(data['country'].unique())}")
        return data
    except FileNotFoundError:
        print(f" Could not find {PATHS['data']}")
        print("Please run exploratory_data_analysis.ipynb first")
        return None

def create_dataloaders(data):
    """Create train/val/test DataLoaders using your existing utils"""
    print("\n Creating DataLoaders...")
    
    # Use your prepare_data function from utils.py
    train_loader, val_loader, test_loader = prepare_data(
    raw_data=data,
    features=DATA_PARAMS['feature_columns'],  # Multiple input features
    target=DATA_PARAMS['target_column'],      # Single target
    window_size=DATA_PARAMS['window_size'],
    batch_size=TRAINING_PARAMS['batch_size']
)
    
    print("DataLoaders created!")
    return train_loader, val_loader, test_loader

def train_models(train_loader, val_loader, test_loader):
    """Train both RNN and Theta models"""
    
    # Train RNN
    print("\n Training RNN Model...")
    rnn_trainer = create_rnn_trainer()
    rnn_results = rnn_trainer.train(train_loader, val_loader, TRAINING_PARAMS['epochs'])
    
    # Train Theta  
    print("\n Training Theta Model...")
    theta_trainer = create_theta_trainer()
    theta_predictions, theta_targets, theta_loss = theta_trainer.train_and_predict(test_loader, "Test")
    
    return rnn_trainer, theta_trainer, rnn_results, (theta_predictions, theta_targets)

def main():
    """Main pipeline execution"""
    print(" COVID-19 FORECASTING PIPELINE")
    print("Statistical Models vs Deep Learning Comparison")
    print("=" * 60)
    
    # Step 1: Load data
    data = load_data()
    if data is None:
        return
    
    # Step 2: Create DataLoaders using your utils
    train_loader, val_loader, test_loader = create_dataloaders(data)
    
    # Step 3: Train models
    rnn_trainer, theta_trainer, rnn_results, theta_results = train_models(
        train_loader, val_loader, test_loader
    )
    
    # Step 4: Evaluate and compare models
    print("\n Evaluating and Comparing Models...")
    evaluator = ModelEvaluator()
    comparison_results = evaluator.compare_models(
        rnn_trainer, theta_trainer, rnn_results, theta_results, test_loader
    )
    
    # Step 5: Final summary
    print("\n PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f" Winner: {comparison_results['winner']}")
    print(f"Results saved in: {PATHS['results']['plots']} and {PATHS['results']['reports']}")

if __name__ == "__main__":
    main()