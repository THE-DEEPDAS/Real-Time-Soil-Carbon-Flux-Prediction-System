import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.append("../preprocessing")
from data_processor import DataProcessor

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.transformer(x)
        # Take the last sequence element for prediction
        x = x[:, -1, :]
        return self.linear(x)

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TimeSeriesTransformer(
            input_dim=3,  # co2, ph, moisture
            num_heads=self.config['transformer']['num_heads'],
            num_layers=self.config['transformer']['num_layers']
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
    def train(self, sequences, labels, epochs=None):
        if epochs is None:
            epochs = self.config['epochs']
            
        sequences = torch.FloatTensor(sequences).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(sequences)
            loss = self.criterion(outputs.squeeze(), labels)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            return self.model(sequence).cpu().numpy()
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    
    # Process data
    processor = DataProcessor(config_path)
    sequences, labels = processor.process_sensor_data("../../data/raw/sensor_data.csv")
    
    # Train model
    trainer = ModelTrainer(config_path)
    trainer.train(sequences, labels)
    
    # Save model
    trainer.save_model("../../models/time_series/model.pth")