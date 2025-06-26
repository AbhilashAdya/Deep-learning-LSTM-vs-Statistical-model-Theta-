"""
Model Evaluation and Comparison for COVID-19 Forecasting
Handles all evaluation, metrics calculation, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import PATHS

# Simple plot configuration
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'colors': {
        'rnn': '#2E86AB',           # Blue for RNN
        'theta': '#A23B72',         # Purple for Theta
        'actual': '#F18F01',        # Orange for actual values
        'train': '#2E8B57',         # Sea green for training
        'val': '#FF6347',           # Tomato for validation
        'test': '#4682B4'           # Steel blue for test
    }
}

class ModelEvaluator:
    """
    Handles evaluation and comparison of RNN and Theta models
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, predictions, targets, model_name):
        """Calculate comprehensive metrics for a model"""
        
        # Flatten arrays for calculation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_flat, pred_flat)
        
        # Calculate MAPE (handle division by zero)
        mask = target_flat != 0
        mape = np.mean(np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])) * 100
        
        # Calculate R-squared
        r2 = r2_score(target_flat, pred_flat)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
        
        print(f"\n{model_name} Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_curves(self, rnn_results):
        """Plot RNN training and validation curves"""
        
        plt.figure(figsize=PLOT_CONFIG['figsize'])
        
        epochs = range(1, len(rnn_results['train_losses']) + 1)
        
        plt.plot(epochs, rnn_results['train_losses'], 
                label='Training Loss', color=PLOT_CONFIG['colors']['train'], linewidth=2)
        plt.plot(epochs, rnn_results['val_losses'], 
                label='Validation Loss', color=PLOT_CONFIG['colors']['val'], linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('RNN Training Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        try:
            plt.savefig(f"{PATHS['results']['plots']}training_curves.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        except:
            plt.savefig("training_curves.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_comparison(self, rnn_pred, rnn_targets, theta_pred, theta_targets):
        """Create simplified prediction comparison plots"""
        
        # Take smaller sample to avoid performance issues
        n_sequences = min(3, len(rnn_pred))
        
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1-3: Sample sequences comparison
        for i in range(n_sequences):
            plt.subplot(2, 2, i+1)
            
            # Plot actual vs predicted (first 10 time steps only for clarity)
            time_steps = range(min(10, len(rnn_targets[i])))
            actual_vals = rnn_targets[i][:len(time_steps)]
            rnn_vals = rnn_pred[i][:len(time_steps)]
            theta_vals = theta_pred[i][:len(time_steps)]
            
            plt.plot(time_steps, actual_vals, 'o-', 
                    label='Actual', color=PLOT_CONFIG['colors']['actual'], linewidth=2, markersize=4)
            plt.plot(time_steps, rnn_vals, 's--', 
                    label='RNN', color=PLOT_CONFIG['colors']['rnn'], alpha=0.8, linewidth=2)
            plt.plot(time_steps, theta_vals, '^--', 
                    label='Theta', color=PLOT_CONFIG['colors']['theta'], alpha=0.8, linewidth=2)
            
            plt.title(f'Sequence {i+1} (First 10 Steps)', fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('COVID Cases (Scaled)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Overall performance scatter (sample data only)
        plt.subplot(2, 2, 4)
        
        # Sample data to avoid memory issues
        sample_size = min(1000, len(rnn_targets.flatten()))
        indices = np.random.choice(len(rnn_targets.flatten()), sample_size, replace=False)
        
        actual_sample = rnn_targets.flatten()[indices]
        rnn_sample = rnn_pred.flatten()[indices]
        theta_sample = theta_pred.flatten()[indices]
        
        plt.scatter(actual_sample, rnn_sample, alpha=0.6, 
                   color=PLOT_CONFIG['colors']['rnn'], s=20, label='RNN')
        plt.scatter(actual_sample, theta_sample, alpha=0.6, 
                   color=PLOT_CONFIG['colors']['theta'], s=20, label='Theta')
        
        # Perfect prediction line
        min_val, max_val = actual_sample.min(), actual_sample.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Performance Comparison (Sample)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        try:
            plt.savefig(f"{PATHS['results']['plots']}predictions_comparison.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        except:
            plt.savefig("predictions_comparison.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        
        plt.show()
        print("Prediction comparison plots created successfully")
    
    def plot_metrics_comparison(self, rnn_metrics, theta_metrics):
        """Create bar chart comparing model metrics"""
        
        metrics_names = list(rnn_metrics.keys())
        rnn_values = list(rnn_metrics.values())
        theta_values = list(theta_metrics.values())
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, rnn_values, width, 
                      label='RNN', color=PLOT_CONFIG['colors']['rnn'], alpha=0.8)
        bars2 = ax.bar(x + width/2, theta_values, width, 
                      label='Theta', color=PLOT_CONFIG['colors']['theta'], alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        
        # Save plot
        try:
            plt.savefig(f"{PATHS['results']['plots']}metrics_comparison.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        except:
            plt.savefig("metrics_comparison.png", dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        
        plt.show()
    
    def determine_winner(self, rnn_metrics, theta_metrics):
        """Determine and announce the winning model"""
        
        print("\n" + "="*50)
        print("FINAL MODEL COMPARISON")
        print("="*50)
        
        # Compare on primary metric (MSE)
        rnn_mse = rnn_metrics['MSE']
        theta_mse = theta_metrics['MSE']
        
        if rnn_mse < theta_mse:
            winner = "RNN Model"
            improvement = ((theta_mse - rnn_mse) / theta_mse) * 100
            print(f"Winner: {winner}")
            print(f"   Improvement: {improvement:.2f}% better MSE")
        else:
            winner = "Theta Model"
            improvement = ((rnn_mse - theta_mse) / rnn_mse) * 100
            print(f"Winner: {winner}")
            print(f"   Improvement: {improvement:.2f}% better MSE")
        
        # Detailed comparison
        print(f"\nDetailed Comparison:")
        print(f"{'Metric':<8} {'RNN':<12} {'Theta':<12} {'Better':<10}")
        print("-" * 45)
        
        for metric in rnn_metrics.keys():
            rnn_val = rnn_metrics[metric]
            theta_val = theta_metrics[metric]
            
            if metric == 'R2':  # Higher is better for R2
                better = "RNN" if rnn_val > theta_val else "Theta"
            else:  # Lower is better for error metrics
                better = "RNN" if rnn_val < theta_val else "Theta"
            
            print(f"{metric:<8} {rnn_val:<12.4f} {theta_val:<12.4f} {better:<10}")
        
        return winner
    
    def save_results(self, rnn_metrics, theta_metrics, winner):
        """Save evaluation results to files"""
        
        print("\nSaving evaluation results...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': ['RNN', 'Theta'],
            'MSE': [rnn_metrics['MSE'], theta_metrics['MSE']],
            'RMSE': [rnn_metrics['RMSE'], theta_metrics['RMSE']],
            'MAE': [rnn_metrics['MAE'], theta_metrics['MAE']],
            'MAPE': [rnn_metrics['MAPE'], theta_metrics['MAPE']],
            'R2': [rnn_metrics['R2'], theta_metrics['R2']]
        })
        
        # Save to CSV
        try:
            comparison_df.to_csv(f"{PATHS['results']['reports']}model_comparison.csv", index=False)
            pd.DataFrame([summary]).to_csv(f"{PATHS['results']['reports']}evaluation_summary.csv", index=False)
        except:
            comparison_df.to_csv("model_comparison.csv", index=False)
            pd.DataFrame([summary]).to_csv("evaluation_summary.csv", index=False)
        
        print("Results saved to CSV files") #Save summary report
        summary = {
            'winner': winner,
            'rnn_mse': rnn_metrics['MSE'],
            'theta_mse': theta_metrics['MSE'],
            'improvement_percent': abs(rnn_metrics['MSE'] - theta_metrics['MSE']) / max(rnn_metrics['MSE'], theta_metrics['MSE']) * 100
        }
        
        pd.DataFrame([summary]).to_csv(f"{PATHS['results']['reports']}evaluation_summary.csv", index=False)
        
        print("✅ Results saved to CSV files")
    
    def compare_models(self, rnn_trainer, theta_trainer, rnn_results, theta_results, test_loader):
        """
        Main function to compare RNN and Theta models
        """
        print("\nCOMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Get predictions
        print("Getting model predictions...")
        rnn_pred, rnn_targets = rnn_trainer.predict(test_loader)
        theta_pred, theta_targets = theta_results[0], theta_results[1]  # predictions, targets
        
        print(f"RNN predictions shape: {rnn_pred.shape}")
        print(f"Theta predictions shape: {theta_pred.shape}")
        
        # Calculate metrics
        rnn_metrics = self.calculate_metrics(rnn_pred, rnn_targets, "RNN")
        theta_metrics = self.calculate_metrics(theta_pred, theta_targets, "Theta")
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_training_curves(rnn_results)
        self.plot_predictions_comparison(rnn_pred, rnn_targets, theta_pred, theta_targets)
        self.plot_metrics_comparison(rnn_metrics, theta_metrics)
        
        # Determine winner
        winner = self.determine_winner(rnn_metrics, theta_metrics)
        
        # Save results
        self.save_results(rnn_metrics, theta_metrics, winner)
        
        print("\nModel evaluation completed!")
        return {
            'rnn_metrics': rnn_metrics,
            'theta_metrics': theta_metrics,
            'winner': winner
        }