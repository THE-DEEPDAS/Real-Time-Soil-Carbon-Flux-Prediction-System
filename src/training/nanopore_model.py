import torch
import torch.nn as nn
import yaml
from pathlib import Path

class NanoporeCNN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate size after convolutions
        self.flatten_size = (input_size // 8) * 128
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NanoporeTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NanoporeCNN(
            input_size=self.config['cnn']['input_size'],
            num_channels=self.config['cnn']['num_channels'],
            num_classes=self.config['cnn']['num_classes']
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
            loss = self.criterion(outputs, labels)
            
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
    trainer = NanoporeTrainer(config_path)
    # Add training data processing and training here when nanopore data is available