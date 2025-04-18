import torch
import torch.nn as nn
import yaml
from pathlib import Path
from .model_trainer import TimeSeriesTransformer
from .nanopore_model import NanoporeCNN

class FusedModel(nn.Module):
    def __init__(self, time_series_dim, nanopore_input_size, nanopore_channels):
        super().__init__()
        
        # Load config
        with open("../../configs/simulation_config.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Time series branch
        self.time_series_model = TimeSeriesTransformer(
            input_dim=time_series_dim,
            num_heads=self.config['transformer']['num_heads'],
            num_layers=self.config['transformer']['num_layers']
        )
        
        # Nanopore branch
        self.nanopore_model = NanoporeCNN(
            input_size=nanopore_input_size,
            num_channels=nanopore_channels,
            num_classes=self.config['cnn']['num_classes']
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),  # Combines outputs from both models
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, time_series_data, nanopore_data):
        # Process time series data
        time_series_output = self.time_series_model(time_series_data)
        
        # Process nanopore data
        nanopore_output = self.nanopore_model(nanopore_data)
        
        # Combine outputs
        combined = torch.cat((time_series_output, nanopore_output), dim=1)
        
        # Final prediction
        return self.fusion(combined)

class FusedModelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FusedModel(
            time_series_dim=3,  # co2, ph, moisture
            nanopore_input_size=self.config['cnn']['input_size'],
            nanopore_channels=self.config['cnn']['num_channels']
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
    
    def train(self, time_series_data, nanopore_data, labels, epochs=None):
        if epochs is None:
            epochs = self.config['epochs']
        
        time_series_data = torch.FloatTensor(time_series_data).to(self.device)
        nanopore_data = torch.FloatTensor(nanopore_data).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(time_series_data, nanopore_data)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, time_series_data, nanopore_data):
        self.model.eval()
        with torch.no_grad():
            time_series_data = torch.FloatTensor(time_series_data).unsqueeze(0).to(self.device)
            nanopore_data = torch.FloatTensor(nanopore_data).unsqueeze(0).to(self.device)
            return self.model(time_series_data, nanopore_data).cpu().numpy()
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def load_pretrained_models(self, time_series_path, nanopore_path):
        """Load pretrained weights for individual models"""
        self.model.time_series_model.load_state_dict(torch.load(time_series_path))
        self.model.nanopore_model.load_state_dict(torch.load(nanopore_path))

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    trainer = FusedModelTrainer(config_path)
    # Add training data processing and training here when both types of data are available