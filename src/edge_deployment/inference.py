import sys
import time
from pathlib import Path
import yaml
sys.path.append("../data_pipeline")
sys.path.append("../preprocessing")
sys.path.append("../training")

from sensor_simulator import SensorSimulator
from data_processor import DataProcessor
from model_trainer import ModelTrainer

class EdgeInference:
    def __init__(self, config_path):
        self.config_path = config_path
        self.simulator = SensorSimulator(config_path)
        self.processor = DataProcessor(config_path)
        self.model = ModelTrainer(config_path)
        
        # Load trained model
        model_path = Path("../../models/time_series/model.pth")
        if model_path.exists():
            self.model.load_model(str(model_path))
        else:
            raise FileNotFoundError("Trained model not found. Please run training first.")
    
    def run_inference_loop(self, interval_seconds=300):
        """Run continuous inference loop, simulating real-time sensor data processing."""
        print("Starting inference loop...")
        
        while True:
            try:
                # Generate latest sensor data (1 hour window)
                data = self.simulator.generate_sensor_data(duration_hours=1)
                
                # Save raw data
                raw_data_path = "../../data/raw/latest_sensor_data.csv"
                self.simulator.save_data(data, raw_data_path)
                
                # Process data
                sequences, _ = self.processor.process_sensor_data(raw_data_path)
                
                # Get prediction for latest sequence
                if len(sequences) > 0:
                    latest_sequence = sequences[-1]
                    prediction = self.model.predict(latest_sequence)
                    
                    # Convert scaled prediction back to real CO2 flux value
                    co2_flux = self.processor.inverse_transform_co2(prediction)[0][0]
                    
                    print(f"Predicted CO2 Flux: {co2_flux:.2f} grams CO2/m²/hour")
                    
                    # Check if above threshold
                    threshold = self.config['co2_flux_alert']
                    if co2_flux > threshold:
                        print(f"ALERT: CO2 flux ({co2_flux:.2f}) exceeds threshold ({threshold})!")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nStopping inference loop...")
                break
            except Exception as e:
                print(f"Error in inference loop: {e}")
                time.sleep(interval_seconds)

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    edge_system = EdgeInference(config_path)
    edge_system.run_inference_loop()