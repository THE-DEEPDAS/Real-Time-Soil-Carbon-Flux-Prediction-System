import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path

class SensorSimulator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sensors = self.config['sensors']
        self.sampling_rate = self.config['sampling_rate']

    def generate_sensor_data(self, duration_hours=24):
        timestamps = []
        co2_values = []
        ph_values = []
        moisture_values = []
        
        start_time = datetime.now()
        num_samples = int((duration_hours * 3600) / self.sampling_rate)
        
        for i in range(num_samples):
            current_time = start_time + timedelta(seconds=i * self.sampling_rate)
            timestamps.append(current_time)
            
            # Generate periodic patterns with noise
            time_factor = 2 * np.pi * i / num_samples
            
            # CO2 simulation with daily cycle
            co2_base = np.sin(time_factor) * 0.5 + 0.5
            co2 = self._scale_with_noise('co2', co2_base)
            co2_values.append(co2)
            
            # pH simulation
            ph_base = np.sin(time_factor * 0.5) * 0.3 + 0.5
            ph = self._scale_with_noise('ph', ph_base)
            ph_values.append(ph)
            
            # Moisture simulation
            moisture_base = np.sin(time_factor * 0.3) * 0.4 + 0.5
            moisture = self._scale_with_noise('moisture', moisture_base)
            moisture_values.append(moisture)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'co2_ppm': co2_values,
            'ph': ph_values,
            'moisture_percent': moisture_values
        })
        
        return df
    
    def _scale_with_noise(self, sensor_type, base_value):
        sensor_config = self.sensors[sensor_type]
        min_val = sensor_config['min_value']
        max_val = sensor_config['max_value']
        noise_level = sensor_config['noise_level']
        
        # Scale base value to sensor range
        scaled_value = min_val + (max_val - min_val) * base_value
        
        # Add noise
        noise = np.random.normal(0, noise_level * (max_val - min_val))
        value_with_noise = scaled_value + noise
        
        # Clip to valid range
        return np.clip(value_with_noise, min_val, max_val)

    def save_data(self, df, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    simulator = SensorSimulator(config_path)
    data = simulator.generate_sensor_data(duration_hours=24)
    simulator.save_data(data, "../../data/raw/sensor_data.csv")