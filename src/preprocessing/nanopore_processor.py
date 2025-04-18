import numpy as np
from scipy import signal
from pathlib import Path
import h5py
import logging
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NanoporeProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sampling_rate = self.config['nanopore']['sampling_rate']
        self.scaler = StandardScaler()
    
    def process_raw_signal(self, signal_data):
        """Process raw nanopore signal data"""
        # Apply baseline correction
        corrected = self._correct_baseline(signal_data)
        
        # Denoise signal
        denoised = self._denoise_signal(corrected)
        
        # Normalize
        normalized = self.scaler.fit_transform(denoised.reshape(-1, 1)).reshape(-1)
        
        return normalized
    
    def _correct_baseline(self, signal_data, window_size=1000):
        """Correct baseline drift using rolling median"""
        baseline = signal.medfilt(signal_data, window_size)
        return signal_data - baseline
    
    def _denoise_signal(self, signal_data):
        """Denoise signal using Savitzky-Golay filter"""
        return signal.savgol_filter(signal_data, window_length=15, polyorder=3)
    
    def segment_signal(self, signal_data, segment_length=4000):
        """Segment signal into fixed-length chunks"""
        num_segments = len(signal_data) // segment_length
        segments = []
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = signal_data[start:end]
            segments.append(segment)
        
        return np.array(segments)
    
    def extract_features(self, signal_segment):
        """Extract relevant features from signal segment"""
        features = {
            'mean': np.mean(signal_segment),
            'std': np.std(signal_segment),
            'median': np.median(signal_segment),
            'max': np.max(signal_segment),
            'min': np.min(signal_segment),
            'range': np.ptp(signal_segment),
            'rms': np.sqrt(np.mean(np.square(signal_segment))),
            'zero_crossings': np.sum(np.diff(np.signbit(signal_segment)))
        }
        return features
    
    def process_fast5_file(self, file_path):
        """Process a FAST5 file containing nanopore data"""
        try:
            with h5py.File(file_path, 'r') as f:
                # This structure might need to be adjusted based on your FAST5 file format
                raw_data = f['Raw/Reads/Read_0/Signal'][:]
                
                # Process signal
                processed_signal = self.process_raw_signal(raw_data)
                
                # Segment signal
                segments = self.segment_signal(processed_signal)
                
                # Extract features for each segment
                features = []
                for segment in segments:
                    segment_features = self.extract_features(segment)
                    features.append(segment_features)
                
                return segments, features
                
        except Exception as e:
            logger.error(f"Error processing FAST5 file {file_path}: {e}")
            return None, None
    
    def process_directory(self, directory_path):
        """Process all FAST5 files in a directory"""
        directory = Path(directory_path)
        processed_data = []
        
        for fast5_file in directory.glob("*.fast5"):
            segments, features = self.process_fast5_file(fast5_file)
            if segments is not None:
                processed_data.append({
                    'file': fast5_file.name,
                    'segments': segments,
                    'features': features
                })
        
        return processed_data
    
    def save_processed_data(self, processed_data, output_path):
        """Save processed nanopore data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, processed_data)
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    config_path = "../../configs/simulation_config.yaml"
    processor = NanoporeProcessor(config_path)
    
    # Example usage with simulated data
    simulated_signal = np.random.normal(0, 1, 20000)
    processed = processor.process_raw_signal(simulated_signal)
    segments = processor.segment_signal(processed)
    
    # Save example processed data
    processor.save_processed_data(
        segments,
        "../../data/processed/example_nanopore_data.npy"
    )