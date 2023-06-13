"""
This file acts on the extracted time series coordinates from the video using mediapipes and extracts features from them using tsfresh.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Define empty lists
list_0 = []
list_1 = []
y = []

# Read npy files and append to lists
for file in os.listdir('dataset_creation/5_features_extracted/0'):
    if file.endswith('.npy'):
        data = np.load(os.path.join('dataset_creation/5_features_extracted/0', file))
        data_df = pd.DataFrame(data)
        list_0.append(data_df)
        y.append(0)
        
for file in os.listdir('dataset_creation/5_features_extracted/1'):
    if file.endswith('.npy'):
        data = np.load(os.path.join('dataset_creation/5_features_extracted/1', file))
        data_df = pd.DataFrame(data)
        list_1.append(data_df)
        y.append(1)

# Convert lists to dataframes
id = 0
df_list = []
for i, data_list in enumerate([list_0, list_1]):
    for j, data_df in enumerate(data_list):
        data_df['target'] = i
        data_df['id'] = id
        id += 1
        df_list.append(data_df)

df = pd.concat(df_list, ignore_index=True)

# Plot the time series
# df[df['id'] == 3].drop(['id','target'],axis=1).plot(subplots=False, sharex=True, figsize=(10,10))

# Extract features using tsfresh
# extraction_settings = MinimalFCParameters()
X = extract_features(df.drop('target', axis=1), column_id='id', column_sort=None,
                     # default_fc_parameters=extraction_settings,
                     # we impute = remove all NaN features automatically
                     impute_function=impute,
                     n_jobs=1)

y = pd.Series(y)

# Save X and y as pickle files
X.to_pickle('dataset_creation/6_X.pkl')
y.to_pickle('dataset_creation/6_y.pkl')
