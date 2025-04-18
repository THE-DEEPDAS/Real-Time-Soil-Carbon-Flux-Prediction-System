import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
    
    def evaluate_predictions(self, y_true, y_pred, model_name):
        """Evaluate model predictions using multiple metrics"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mae': np.mean(np.abs(y_true - y_pred))
        }
        
        self.metrics_history.append({
            'model_name': model_name,
            'metrics': metrics
        })
        
        logger.info(f"Model: {model_name} - Metrics: {metrics}")
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title, save_name):
        """Plot true vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(title)
        
        plot_path = self.save_dir / f"{save_name}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved prediction plot to {plot_path}")
    
    def plot_residuals(self, y_true, y_pred, title, save_name):
        """Plot prediction residuals"""
        residuals = y_pred - y_true
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30)
        plt.xlabel('Residual Value')
        plt.ylabel('Count')
        plt.title(f"Residual Distribution - {title}")
        
        plot_path = self.save_dir / f"{save_name}_residuals.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved residual plot to {plot_path}")
    
    def save_metrics(self, filename="evaluation_metrics.json"):
        """Save evaluation metrics to file"""
        metrics_path = self.save_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    def compare_models(self, metric='rmse'):
        """Compare different models based on a specific metric"""
        if not self.metrics_history:
            logger.warning("No metrics available for comparison")
            return
        
        models = []
        scores = []
        
        for entry in self.metrics_history:
            models.append(entry['model_name'])
            scores.append(entry['metrics'][metric])
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, scores)
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.xticks(rotation=45)
        plt.ylabel(metric.upper())
        
        plot_path = self.save_dir / f"model_comparison_{metric}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {plot_path}")

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator("../../models/evaluation")
    
    # Simulate some predictions
    y_true = np.random.normal(0, 1, 100)
    y_pred = y_true + np.random.normal(0, 0.2, 100)
    
    # Evaluate and plot
    metrics = evaluator.evaluate_predictions(y_true, y_pred, "test_model")
    evaluator.plot_predictions(y_true, y_pred, "Test Predictions", "test_predictions")
    evaluator.plot_residuals(y_true, y_pred, "Test Model", "test_model")
    evaluator.save_metrics()