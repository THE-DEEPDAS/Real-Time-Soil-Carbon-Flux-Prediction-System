import logging
import time
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
from threading import Thread, Event
import queue

class MonitoringSystem:
    def __init__(self, config_path, log_dir="../../data/logs"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Alert thresholds
        self.co2_threshold = self.config['co2_flux_alert']
        
        # Initialize data queues for real-time plotting
        self.data_queue = queue.Queue()
        self.stop_event = Event()
        
        # Metrics history
        self.metrics_history = []
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = self.log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.logger.info("Starting monitoring system...")
        
        # Start real-time plotting thread
        self.plot_thread = Thread(target=self._plot_real_time)
        self.plot_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.stop_event.set()
        if hasattr(self, 'plot_thread'):
            self.plot_thread.join()
        self.logger.info("Monitoring system stopped")
    
    def process_measurement(self, measurement):
        """Process new measurement and check for alerts"""
        timestamp = datetime.now()
        
        # Add to history
        self.metrics_history.append({
            'timestamp': timestamp,
            'data': measurement
        })
        
        # Add to plotting queue
        self.data_queue.put((timestamp, measurement))
        
        # Check alerts
        self._check_alerts(measurement)
        
        # Log measurement
        self._log_measurement(timestamp, measurement)
    
    def _check_alerts(self, measurement):
        """Check for alert conditions"""
        if measurement.get('co2_flux', 0) > self.co2_threshold:
            self.logger.warning(
                f"HIGH CO2 FLUX ALERT: {measurement['co2_flux']:.2f} g/m²/hour "
                f"(threshold: {self.co2_threshold})"
            )
    
    def _log_measurement(self, timestamp, measurement):
        """Log measurement to file"""
        log_file = self.log_dir / f"measurements_{timestamp.strftime('%Y%m%d')}.json"
        
        entry = {
            'timestamp': timestamp.isoformat(),
            'measurement': measurement
        }
        
        with open(log_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
    
    def _plot_real_time(self):
        """Real-time plotting of measurements"""
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        timestamps = []
        co2_values = []
        ph_values = []
        moisture_values = []
        
        while not self.stop_event.is_set():
            try:
                timestamp, measurement = self.data_queue.get(timeout=1)
                
                timestamps.append(timestamp)
                co2_values.append(measurement.get('co2_flux', 0))
                ph_values.append(measurement.get('ph', 0))
                moisture_values.append(measurement.get('moisture_percent', 0))
                
                # Keep last 100 points
                if len(timestamps) > 100:
                    timestamps.pop(0)
                    co2_values.pop(0)
                    ph_values.pop(0)
                    moisture_values.pop(0)
                
                # Update plots
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                ax1.plot(timestamps, co2_values)
                ax1.set_ylabel('CO2 Flux (g/m²/hour)')
                ax1.axhline(y=self.co2_threshold, color='r', linestyle='--')
                
                ax2.plot(timestamps, ph_values)
                ax2.set_ylabel('pH')
                
                ax3.plot(timestamps, moisture_values)
                ax3.set_ylabel('Moisture (%)')
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in plotting: {e}")
    
    def save_metrics_summary(self):
        """Save summary statistics of monitored metrics"""
        if not self.metrics_history:
            return
        
        summary = {
            'start_time': self.metrics_history[0]['timestamp'].isoformat(),
            'end_time': self.metrics_history[-1]['timestamp'].isoformat(),
            'total_measurements': len(self.metrics_history),
            'statistics': {
                'co2_flux': {
                    'mean': np.mean([m['data'].get('co2_flux', 0) for m in self.metrics_history]),
                    'max': np.max([m['data'].get('co2_flux', 0) for m in self.metrics_history]),
                    'min': np.min([m['data'].get('co2_flux', 0) for m in self.metrics_history])
                },
                'ph': {
                    'mean': np.mean([m['data'].get('ph', 0) for m in self.metrics_history]),
                    'max': np.max([m['data'].get('ph', 0) for m in self.metrics_history]),
                    'min': np.min([m['data'].get('ph', 0) for m in self.metrics_history])
                },
                'moisture': {
                    'mean': np.mean([m['data'].get('moisture_percent', 0) for m in self.metrics_history]),
                    'max': np.max([m['data'].get('moisture_percent', 0) for m in self.metrics_history]),
                    'min': np.min([m['data'].get('moisture_percent', 0) for m in self.metrics_history])
                }
            }
        }
        
        summary_file = self.log_dir / 'monitoring_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Saved metrics summary to {summary_file}")

if __name__ == "__main__":
    # Example usage
    config_path = "../../configs/simulation_config.yaml"
    monitor = MonitoringSystem(config_path)
    
    try:
        monitor.start_monitoring()
        
        # Simulate some measurements
        for _ in range(50):
            measurement = {
                'co2_flux': np.random.normal(2.0, 0.5),
                'ph': np.random.normal(6.5, 0.2),
                'moisture_percent': np.random.normal(50, 5)
            }
            monitor.process_measurement(measurement)
            time.sleep(1)
            
    finally:
        monitor.stop_monitoring()
        monitor.save_metrics_summary()