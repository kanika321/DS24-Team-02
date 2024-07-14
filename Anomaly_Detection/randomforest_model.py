import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def get_all_parquet_paths(root_dir, file_type):
    parquet_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet') and file_type in file:
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def load_parquet_files(file_paths):
    data = []
    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            data.append(df.values.flatten())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return data

def downsample_data(data, original_freq, target_freq):
    downsampled_data = []
    for file in data:
        downsampled_file = signal.resample_poly(file, target_freq, original_freq)
        downsampled_data.append(downsampled_file)
    return downsampled_data

def extract_features(data):
    features = []
    for file in data:
        mean = np.mean(file)
        std = np.std(file)
        skewness = pd.Series(file).skew()
        kurtosis = pd.Series(file).kurt()
        peak_to_peak = np.ptp(file)
        
        # FFT for frequency domain features
        fft_values = np.fft.fft(file)
        fft_magnitudes = np.abs(fft_values)
        dominant_freq = np.argmax(fft_magnitudes)
        spectral_centroid = np.sum(np.arange(len(fft_magnitudes)) * fft_magnitudes) / np.sum(fft_magnitudes)
        
        features.append([mean, std, skewness, kurtosis, peak_to_peak, dominant_freq, spectral_centroid])
    return np.array(features)

# Define root directories
nok_root_dir = r"../Data/NOK_Measurements_zipped/NOK_Measurements/NOK_Measurements"
ok_root_dir = r"../Data/OK_Measurements_zipped/OK_Measurements"

# Get all 100KHz parquet file paths
nok_100khz_files = get_all_parquet_paths(nok_root_dir, 'Sampling100KHz')
ok_100khz_files = get_all_parquet_paths(ok_root_dir, 'Sampling100KHz')

# Load spindle data (assuming spindle data is sampled at 100 KHz)
nok_spindle_data = load_parquet_files(nok_100khz_files)
ok_spindle_data = load_parquet_files(ok_100khz_files)

# Get all 2MHz parquet file paths
nok_acoustic_files = get_all_parquet_paths(nok_root_dir, 'Sampling2000KHz')
ok_acoustic_files = get_all_parquet_paths(ok_root_dir, 'Sampling2000KHz')

# Load acoustic data (assuming acoustic data is sampled at 2 MHz)
nok_acoustic_data = load_parquet_files(nok_acoustic_files)
ok_acoustic_data = load_parquet_files(ok_acoustic_files)

# Downsample acoustic data to match spindle data frequency
original_freq_acoustic = 2000000  # 2 MHz
target_freq = 100000  # 100 KHz

nok_acoustic_data_downsampled = downsample_data(nok_acoustic_data, original_freq_acoustic, target_freq)
ok_acoustic_data_downsampled = downsample_data(ok_acoustic_data, original_freq_acoustic, target_freq)
print("Downsample Done")

# Preprocess spindle current data
scaler = StandardScaler()

ok_spindle_data = [scaler.fit_transform(file.reshape(-1, 1)).flatten() for file in ok_spindle_data]
nok_spindle_data = [scaler.transform(file.reshape(-1, 1)).flatten() for file in nok_spindle_data]

# Extract features
ok_spindle_features = extract_features(ok_spindle_data)
nok_spindle_features = extract_features(nok_spindle_data)

ok_acoustic_features = extract_features(ok_acoustic_data_downsampled)
nok_acoustic_features = extract_features(nok_acoustic_data_downsampled)

# Combine features for acoustic and spindle data
ok_combined_features = np.hstack((ok_acoustic_features, ok_spindle_features))
nok_combined_features = np.hstack((nok_acoustic_features, nok_spindle_features))

# Generate labels
ok_labels = np.zeros(len(ok_combined_features))
nok_labels = np.ones(len(nok_combined_features))

X = np.vstack((ok_combined_features, nok_combined_features))
y = np.hstack((ok_labels, nok_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
