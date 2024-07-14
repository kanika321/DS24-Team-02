#!/usr/bin/env python3                                                                                                                                                                                                                                                                                                              #!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_all_parquet_paths(root_dir, file_type):
    parquet_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.parquet') and file_type in file:
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def process_file(file_path, label, stat_collector):
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        df['Label'] = label
        stat_collector['rows'] += len(df)
        if not stat_collector['columns']:
            stat_collector['columns'] = df.columns.tolist()
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col not in stat_collector['describe']:
                stat_collector['describe'][col] = []
                stat_collector['hist'][col] = []
            stat_collector['describe'][col].append(df[col].describe())
            stat_collector['hist'][col].extend(df[col].values.tolist())
        stat_collector['time_series'].append(df.select_dtypes(include=['float64', 'int64']).head(5))  # Take a sample for time series
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Define root directories
nok_root_dir = r"../Data/NOK_Measurements_zipped/NOK_Measurements/NOK_Measurements"
ok_root_dir = r"../Data/OK_Measurements_zipped/OK_Measurements"

# Get all 100KHz parquet file paths
nok_100khz_files = get_all_parquet_paths(nok_root_dir, 'Sampling100KHz')

print("NOK 100KHz files:", nok_100khz_files)

if not nok_100khz_files:
    print("No 100KHz parquet files found.")
else:
    print(f"Found {len(nok_100khz_files)} 100KHz parquet files.")

# Initialize statistics collector
stat_collector = {
    'rows': 0,
    'columns': [],
    'describe': {},
    'hist': {},
    'time_series': []
}

# Process each file individually
for file in nok_100khz_files:
    process_file(file, 'Not OK', stat_collector)

# Combine statistics
if stat_collector['rows'] == 0:
    print("No data loaded for Not OK 100KHz files.")
    exit()

# Combine descriptive statistics
print("\nCombined Basic Statistics:")
for col, desc in stat_collector['describe'].items():
    combined_describe = pd.concat(desc, axis=1)
    print(f"\nStatistics for {col}:\n", combined_describe)

# Save Histograms as Image Files
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

print("\nSaving histograms as images...")
num_cols = [col for col in stat_collector['columns'] if col in stat_collector['hist']]
for col in num_cols:
    plt.figure()
    plt.hist(stat_collector['hist'][col], bins=30)
    plt.title(f'Histogram of {col}')
    plt.savefig(f"{output_dir}/histogram_{col}.png")
    plt.close()


# Plot Time Series for a Sample of Columns
print("\nPlotting time series for a sample of columns...")
time_series_sample = pd.concat(stat_collector['time_series'], ignore_index=True)
time_series_sample.plot(subplots=True, figsize=(15, 10), title='Time Series of Sample Columns')
plt.savefig(f"{output_dir}/time_series_sample.png")
plt.close()

# Correlation Analysis
print("\nCorrelation Analysis:")
corr_matrix = time_series_sample.corr()
print(corr_matrix)

# Plot Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Columns')
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()
