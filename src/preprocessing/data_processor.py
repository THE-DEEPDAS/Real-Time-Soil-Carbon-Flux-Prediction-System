import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml

class DataProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sequence_length = self.config['sequence_length']
        self.scalers = {
            'co2_ppm': MinMaxScaler(),
            'ph': MinMaxScaler(),
            'moisture_percent': MinMaxScaler()
        }

    def process_sensor_data(self, data_path):
        # Read data
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Scale features
        scaled_data = df.copy()
        for column in self.scalers.keys():
            scaled_data[column] = self.scalers[column].fit_transform(df[[column]])
        
        # Create sequences for time series model
        sequences = self._create_sequences(scaled_data)
        
        return sequences
    
    def _create_sequences(self, df):
        sequences = []
        labels = []
        
        for i in range(len(df) - self.sequence_length):
            sequence = df.iloc[i:i + self.sequence_length][['co2_ppm', 'ph', 'moisture_percent']].values
            # Use CO2 as target variable (simplified for simulation)
            label = df.iloc[i + self.sequence_length]['co2_ppm']
            
            sequences.append(sequence)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)

    def inverse_transform_co2(self, scaled_values):
        return self.scalers['co2_ppm'].inverse_transform(scaled_values.reshape(-1, 1))

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    processor = DataProcessor(config_path)
    sequences, labels = processor.process_sensor_data("../../data/raw/sensor_data.csv")
    print(f"Created {len(sequences)} sequences of shape {sequences.shape}")