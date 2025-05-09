import os
import mne
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from tqdm import tqdm

from joblib import Parallel, delayed

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../third_parts/microstate_lib/code")
import eeg_recording


class MicrostateKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_microstates=4, max_iter=100, tol=1e-6, 
                 n_init=10, random_state=None, n_jobs=1):
        self.n_microstates = n_microstates
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X, y=None):
        """Fit the microstate model with parallel initialization support"""
        if self.n_jobs != 1 and self.n_init > 1:
            seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_init)
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._single_fit)(X, seed) 
                for seed in seeds
            )
            
            # Find best result
            best_idx = np.argmax([r[1] for r in results])
            self.microstate_maps_ = results[best_idx][0]
            self.labels_ = results[best_idx][2]
            self.inertia_ = results[best_idx][3]
            self.gev_ = results[best_idx][1]
        else:
            self.microstate_maps_, self.labels_, self.inertia_, self.gev_ = self._single_fit(
                X, self.random_state
            )
        
        return self
    
    def _single_fit(self, X, random_state):
        """Single run of microstate clustering"""
        np.random.seed(random_state)
        
        # Initialize using GFP peaks
        gfp = np.std(X, axis=1)
        peak_indices = np.argsort(gfp)[-self.n_microstates:]
        maps = X[peak_indices].copy()
        
        for _ in range(self.max_iter):
            # Assignment with polarity invariance
            correlations = np.abs(X @ maps.T)
            labels = np.argmax(correlations, axis=1)
            
            # Update maps
            new_maps = np.zeros_like(maps)
            for k in range(self.n_microstates):
                cluster_data = X[labels == k]
                if len(cluster_data) > 0:
                    corr = cluster_data @ maps[k]
                    cluster_data[corr < 0] *= -1
                    new_maps[k] = cluster_data.mean(axis=0)
            
            # Check convergence
            if np.linalg.norm(new_maps - maps) < self.tol:
                break
            maps = new_maps
        
        # Calculate metrics
        gev = self._calculate_gev(X, maps)
        print(gev)
        inertia = self._calculate_inertia(X, maps)
        
        return maps, gev, labels, inertia
    
        
    def _calculate_gev(self, X, maps):
        """Correct GEV calculation following Pascual-Marqui (1995)"""
        # Step 1: Assign each sample to the best-fitting microstate (considering polarity)
        # Compute correlation for each sample (accounting for polarity)
        correlations = np.dot(X, maps.T)  # Shape: (n_samples, n_microstates)
        
        # For each sample, find the microstate that maximizes the correlation (sign and magnitude)
        labels = np.argmax(correlations, axis=1)
        microstate_sequence = maps[labels]  # Shape: (n_samples, n_features)
        
        # Step 2: Compute GFP² (global field power squared)
        gfp_sq = np.var(X, axis=1)  # Equivalent to (X ** 2).mean(axis=1) for z-scored data
        
        # Step 3: For each sample, compute (GFP * correlation with assigned map)²
        # Compute norms (magnitude) of microstate_sequence and X
        microstate_norm = np.linalg.norm(microstate_sequence, axis=1)
        X_norm = np.linalg.norm(X, axis=1)
        
        # Normalize the sample vectors (X) and microstate sequences (maps)
        numerator = (np.sum(microstate_sequence * X, axis=1) / (microstate_norm * X_norm)) * gfp_sq
        
        # Step 4: Sum and normalize by total GFP²
        gev = np.sum(numerator) / np.sum(gfp_sq ** 2)
        
        return gev  # Return as percentage
    
    def _calculate_inertia(self, X, maps):
        """Calculate inertia (sum of squared distances)"""
        distances = 1 - np.abs(X @ maps.T)  # Correlation distance
        return np.sum(np.min(distances, axis=1))
    
    def predict(self, X):
        """Predict microstate labels for new data"""
        check_is_fitted(self, 'microstate_maps_')
        correlations = np.abs(X @ self.microstate_maps_.T)
        return np.argmax(correlations, axis=1)

    
# Configure paths
sys.path.extend(["..", "../lib", "../third_parts/microstate_lib/code"])

def setup_args():
    parser = argparse.ArgumentParser(description='EEG Data Preprocessing')
    parser.add_argument("--data_set_path", default="./data/preprocessed_data/",
                      help="Path to development dataset")
    parser.add_argument("--output_path", default="./data/microstates/microstates_dev.npy",
                      help="Output directory for processed data")

    parser.add_argument("--max_samples", default=None)
    parser.add_argument("--n_microstates", default=4)
    parser.add_argument("--batch_size", default=100)
    parser.add_argument("--random_state", default=None)

    return parser.parse_args()

def find_raw_files(base_path, suffix = ".fif"):
    """Recursively find all EDF files with corresponding CSV files."""
    valid_files = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(suffix):
                raw_path = os.path.join(root, file)
                valid_files.append(raw_path)
    return valid_files

selected_channels = ('Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz')

def save_microstates(microstates, output_path, filename="group_microstates.npy"):
    """Save microstate maps to file."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    np.save(output_file, microstates)
    print(f"Saved microstates to {output_file}")

def compute_microstates(data, n_microstates=4, batch_size=100, random_state=42):
    """Compute microstates using mini-batch k-means on z-scored data."""
    # Transpose to time × channels format expected by sklearn
    data = data.T  # now samples × channels

    # Initialize and fit mini-batch k-means
    kmeans = MicrostateKMeans(
        n_microstates=n_microstates,
        #batch_size=batch_size,
        random_state=random_state,
        n_init=3,
        n_jobs=16
        #compute_labels=False
    )
    
    kmeans.fit(data)
    
    # Get microstate maps (cluster centers) and transpose back to channels × microstates
    microstates = kmeans.microstate_maps_.T
    
    # Get results
    print(f"Global GEV: {kmeans.gev_}")
    print("Microstate maps:", kmeans.microstate_maps_.shape)

    return microstates


def load_and_concat_raws(files, selected_channels, max_samples=None):
    """Load and concatenate raw data from multiple files with memory management and return raw object."""
    all_raws = []  # To store the raw objects
    
    total_samples = 0
    
    for file in tqdm(files, desc="Loading files"):
        try:
            raw = mne.io.read_raw_fif(file, preload=True)
            raw.pick_channels(selected_channels)
            
            # Apply bandpass filter
            raw.filter(1, 20, method='iir')
            
            data = raw.get_data()  # channels × time
            
            if max_samples is not None:
                available_samples = data.shape[1]
                take_samples = min(available_samples, max_samples - total_samples)
                if take_samples <= 0:
                    break
                raw.crop(tmin=0, tmax=take_samples / raw.info['sfreq'])  # Crop raw object to the selected sample range
                total_samples += take_samples
            else:
                total_samples += data.shape[1]
            
            all_raws.append(raw)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if not all_raws:
        raise ValueError("No valid data found in any files")
    
    # Concatenate the raw objects along the time axis
    combined_raw = mne.io.RawArray(np.concatenate([raw.get_data() for raw in all_raws], axis=1), all_raws[0].info)
    
    return combined_raw

raws = []

def main():
    args = setup_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # Find and load files
    files = find_raw_files(args.data_set_path, ".fif")
    if not files:
        print("No FIF files found in the specified path.")
        return
    
    print(f"Found {len(files)} FIF files for processing")
    
    # Load and concatenate all data
    try:
        concatenated_raw = load_and_concat_raws(
                files[:1000],
            selected_channels=selected_channels,
            max_samples=args.max_samples
        )
    except ValueError as e:
        print(e)
        return

    # Compute microstates
    recording = eeg_recording.SingleSubjectRecording("0", concatenated_raw)
    recording.run_latent_kmeans(4, use_gfp = True)
    
    print(recording.gev_tot)
    print(recording.latent_maps)

    # Save results
    np.save(args.output_path + f'{recording.gev_tot}', recording.latent_maps)

if __name__ == "__main__":
    main()
