#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_all_parquet_paths(root_dir, keyword):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".parquet") and keyword in file:
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_parquet_files(file_paths):
    data_frames = []
    for file_path in file_paths:
        df = pq.read_table(file_path).to_pandas()
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

# Define root directories (update these paths based on your Google Drive structure)
nok_root_dir = '../Data/NOK_Measurements_zipped/NOK_Measurements/NOK_Measurements'
ok_root_dir = '../Data/OK_Measurements_zipped/OK_Measurements'

print(nok_root_dir)
print(ok_root_dir)

# Get all parquet file paths
nok_100khz_files = get_all_parquet_paths(nok_root_dir, 'Sampling100KHz')
ok_100khz_files = get_all_parquet_paths(ok_root_dir, 'Sampling100KHz')

# Load the data
nok_100khz_data = load_parquet_files(nok_100khz_files)
ok_100khz_data = load_parquet_files(ok_100khz_files)

# Preprocess the data
ok_100khz_data_scaled, scaler = preprocess_data(ok_100khz_data)
nok_100khz_data_scaled = scaler.transform(nok_100khz_data)

# Create a subplot figure
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=('OK Data', 'NOK Data'))

# Add OK data plot
fig.add_trace(go.Scatter(x=np.arange(1000), y=ok_100khz_data_scaled[:1000, 0],
                         mode='lines', name='Irms Grinding (OK)'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=ok_100khz_data_scaled[:1000, 1],
                         mode='lines', name='Spindle Current L1 (OK)'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=ok_100khz_data_scaled[:1000, 2],
                         mode='lines', name='Spindle Current L2 (OK)'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=ok_100khz_data_scaled[:1000, 3],
                         mode='lines', name='Spindle Current L3 (OK)'),
              row=1, col=1)

# Add NOK data plot
fig.add_trace(go.Scatter(x=np.arange(1000), y=nok_100khz_data_scaled[:1000, 0],
                         mode='lines', name='Irms Grinding (NOK)'),
              row=2, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=nok_100khz_data_scaled[:1000, 1],
                         mode='lines', name='Spindle Current L1 (NOK)'),
              row=2, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=nok_100khz_data_scaled[:1000, 2],
                         mode='lines', name='Spindle Current L2 (NOK)'),
              row=2, col=1)

fig.add_trace(go.Scatter(x=np.arange(1000), y=nok_100khz_data_scaled[:1000, 3],
                         mode='lines', name='Spindle Current L3 (NOK)'),
              row=2, col=1)

fig.update_layout(title_text='Interactive Sensor Data Visualization',
                  xaxis_title='Time',
                  yaxis_title='Sensor Readings')

# Save the plot to an HTML file
fig.write_html('interactive_plot.html')
print("Interactive plot saved as interactive_plot.html")

