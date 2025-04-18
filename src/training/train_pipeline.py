import sys
from pathlib import Path
import yaml
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_processor import DataProcessor
from preprocessing.nanopore_processor import NanoporeProcessor
from training.model_trainer import ModelTrainer
from training.nanopore_model import NanoporeTrainer
from training.fused_model import FusedModelTrainer
from training.model_evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize processors
        self.sensor_processor = DataProcessor(config_path)
        self.nanopore_processor = NanoporeProcessor(config_path)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator("../../models/evaluation")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, sensor_data_path, nanopore_data_path=None):
        """Prepare data for training"""
        # Process sensor data
        sensor_sequences, sensor_labels = self.sensor_processor.process_sensor_data(sensor_data_path)
        
        # Process nanopore data if available
        nanopore_sequences = None
        if nanopore_data_path:
            nanopore_data = np.load(nanopore_data_path)
            nanopore_sequences = self.nanopore_processor.process_raw_signal(nanopore_data)
            nanopore_sequences = self.nanopore_processor.segment_signal(nanopore_sequences)
        
        # Split data
        if nanopore_sequences is not None:
            # Ensure same number of samples
            min_samples = min(len(sensor_sequences), len(nanopore_sequences))
            sensor_sequences = sensor_sequences[:min_samples]
            nanopore_sequences = nanopore_sequences[:min_samples]
            sensor_labels = sensor_labels[:min_samples]
        
        # Create train/test split
        if nanopore_sequences is not None:
            (sensor_train, sensor_test, 
             nano_train, nano_test, 
             labels_train, labels_test) = train_test_split(
                sensor_sequences, nanopore_sequences, sensor_labels, 
                test_size=0.2, random_state=42
            )
            return {
                'sensor_train': sensor_train,
                'sensor_test': sensor_test,
                'nano_train': nano_train,
                'nano_test': nano_test,
                'labels_train': labels_train,
                'labels_test': labels_test
            }
        else:
            sensor_train, sensor_test, labels_train, labels_test = train_test_split(
                sensor_sequences, sensor_labels, test_size=0.2, random_state=42
            )
            return {
                'sensor_train': sensor_train,
                'sensor_test': sensor_test,
                'labels_train': labels_train,
                'labels_test': labels_test
            }
    
    def train_sensor_model(self, data):
        """Train time series model on sensor data"""
        logger.info("Training sensor time series model...")
        trainer = ModelTrainer(self.config_path)
        
        # Train model
        trainer.train(data['sensor_train'], data['labels_train'])
        
        # Evaluate
        predictions = []
        for sequence in data['sensor_test']:
            pred = trainer.predict(sequence)
            predictions.append(pred[0])
        
        metrics = self.evaluator.evaluate_predictions(
            data['labels_test'],
            np.array(predictions),
            "sensor_model"
        )
        
        # Save model and plots
        trainer.save_model("../../models/time_series/sensor_model.pth")
        self.evaluator.plot_predictions(
            data['labels_test'],
            predictions,
            "Sensor Model Predictions",
            "sensor_model_predictions"
        )
        
        return trainer, metrics
    
    def train_nanopore_model(self, data):
        """Train CNN model on nanopore data"""
        logger.info("Training nanopore CNN model...")
        trainer = NanoporeTrainer(self.config_path)
        
        # Train model
        trainer.train(data['nano_train'], data['labels_train'])
        
        # Evaluate
        predictions = []
        for sequence in data['nano_test']:
            pred = trainer.predict(sequence)
            predictions.append(pred[0])
        
        metrics = self.evaluator.evaluate_predictions(
            data['labels_test'],
            np.array(predictions),
            "nanopore_model"
        )
        
        # Save model and plots
        trainer.save_model("../../models/nanopore_cnn/nanopore_model.pth")
        self.evaluator.plot_predictions(
            data['labels_test'],
            predictions,
            "Nanopore Model Predictions",
            "nanopore_model_predictions"
        )
        
        return trainer, metrics
    
    def train_fused_model(self, data):
        """Train fused model combining sensor and nanopore data"""
        logger.info("Training fused model...")
        trainer = FusedModelTrainer(self.config_path)
        
        # Train model
        trainer.train(
            data['sensor_train'],
            data['nano_train'],
            data['labels_train']
        )
        
        # Evaluate
        predictions = []
        for i in range(len(data['sensor_test'])):
            pred = trainer.predict(
                data['sensor_test'][i:i+1],
                data['nano_test'][i:i+1]
            )
            predictions.append(pred[0])
        
        metrics = self.evaluator.evaluate_predictions(
            data['labels_test'],
            np.array(predictions),
            "fused_model"
        )
        
        # Save model and plots
        trainer.save_model("../../models/fused_model/fused_model.pth")
        self.evaluator.plot_predictions(
            data['labels_test'],
            predictions,
            "Fused Model Predictions",
            "fused_model_predictions"
        )
        
        return trainer, metrics
    
    def run_full_pipeline(self, sensor_data_path, nanopore_data_path=None):
        """Run complete training pipeline"""
        # Prepare data
        data = self.prepare_data(sensor_data_path, nanopore_data_path)
        
        # Train individual models
        sensor_model, sensor_metrics = self.train_sensor_model(data)
        logger.info(f"Sensor model metrics: {sensor_metrics}")
        
        if nanopore_data_path:
            nanopore_model, nanopore_metrics = self.train_nanopore_model(data)
            logger.info(f"Nanopore model metrics: {nanopore_metrics}")
            
            # Train fused model
            fused_model, fused_metrics = self.train_fused_model(data)
            logger.info(f"Fused model metrics: {fused_metrics}")
        
        # Compare models
        self.evaluator.compare_models()
        self.evaluator.save_metrics()
        
        logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    pipeline = TrainingPipeline(config_path)
    
    # Example usage with simulated data
    pipeline.run_full_pipeline(
        "../../data/raw/sensor_data.csv",
        "../../data/processed/example_nanopore_data.npy"
    )