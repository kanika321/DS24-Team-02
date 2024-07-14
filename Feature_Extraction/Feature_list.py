import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Function to get all parquet file paths
def get_all_parquet_paths(root_dir, file_type):
    parquet_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet') and file_type in file:
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# Function to load parquet files
def load_parquet_files(file_paths):
    data = []
    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            data.append(df['AEKi_rate2000000_clipping0_batch0'].values.flatten())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return data

# Function to extract features from a segment of data
def extract_features(segment, sampling_rate=2e6):
    # Time domain features
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    skewness_val = skew(segment)
    kurtosis_val = kurtosis(segment)
    ptp_val = np.ptp(segment)
    
    # Frequency domain features
    fft_vals = fft(segment)
    fft_magnitudes = np.abs(fft_vals)
    freqs = np.fft.fftfreq(len(segment), 1/sampling_rate)
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(fft_magnitudes)
    dominant_freq = freqs[dominant_freq_idx]
    
    # Spectral centroid
    spectral_centroid = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)
    
    # FFT magnitude summary statistics
    fft_mean = np.mean(fft_magnitudes)
    fft_var = np.var(fft_magnitudes)
    
    # Include the most significant FFT magnitudes (e.g., top 5)
    top_n = 5
    top_indices = np.argsort(fft_magnitudes)[-top_n:]
    top_fft_magnitudes = fft_magnitudes[top_indices]
    
    # Create a dictionary with features
    features = {
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness_val,
	'kurtosis': kurtosis_val,
        'peak_to_peak': ptp_val,
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'fft_mean': fft_mean,
        'fft_var': fft_var
    }
    
    # Add top FFT magnitudes to the features dictionary
    for i, magnitude in enumerate(top_fft_magnitudes):
        features[f'top_fft_magnitude_{i+1}'] = magnitude
    
    return features

# Define root directories
nok_root_dir = r"../Data/NOK_Measurements_zipped/NOK_Measurements/NOK_Measurements"
ok_root_dir = r"../Data/OK_Measurements_zipped/OK_Measurements"

# Get all 2MHz parquet file paths
nok_acoustic_files = get_all_parquet_paths(nok_root_dir, 'Sampling2000KHz')
ok_acoustic_files = get_all_parquet_paths(ok_root_dir, 'Sampling2000KHz')

# Initialize an empty list to store features
all_features_list = []

# Define the chunk size (number of samples per chunk)
chunk_size = int(2 * 10**6)  # Example: 1 second worth of data

# Process NOK acoustic files
for file_path in nok_acoustic_files:
    # Load the Parquet file
    try:
        data = pd.read_parquet(file_path)
        print("read file", file_path)
        column_data = data['AEKi_rate2000000_clipping0_batch0']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue
    
    if not column_data.empty:
        features = extract_features(column_data.values.flatten())
        # Add file identifier to features
        features['file'] = file_path
        features['type'] = 'NOK'
        all_features_list.append(features)

print("Processed all NOK files")


"""     # Process the data in chunks
    num_chunks = int(np.ceil(len(column_data) / chunk_size))
    for i in range(num_chunks):
        chunk = column_data.iloc[i*chunk_size:(i+1)*chunk_size]
        if not chunk.empty:
            features = extract_features(chunk.values.flatten())
            # Add file identifier to features
            features['file'] = file_path
            features['type'] = 'NOK'
            all_features_list.append(features) """



# Process OK acoustic files
for file_path in ok_acoustic_files:
    # Load the Parquet file
    try:
        data = pd.read_parquet(file_path)
        print("read file", file_path)
        column_data = data['AEKi_rate2000000_clipping0_batch0']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue
    

    # Process the entire data
    if not column_data.empty:
        features = extract_features(column_data.values.flatten())
        # Add file identifier to features
        features['file'] = file_path
        features['type'] = 'OK'
        all_features_list.append(features)

print("Processed all OK files")

"""     # Process the data in chunks
    num_chunks = int(np.ceil(len(column_data) / chunk_size))
    for i in range(num_chunks):
        chunk = column_data.iloc[i*chunk_size:(i+1)*chunk_size]
        if not chunk.empty:
            features = extract_features(chunk.values.flatten())
            # Add file identifier to features
            features['file'] = file_path
            features['type'] = 'OK'
            all_features_list.append(features) """

# Convert the list of feature dictionaries to a DataFrame
all_features_df = pd.DataFrame(all_features_list)

# Define the output CSV file path
output_csv_path = r"../torch-condor-template/feature_list1.csv"

# Save the features dataframe to a CSV file
all_features_df.to_csv(output_csv_path, index=False)

print(f"Features successfully saved to {output_csv_path}")
