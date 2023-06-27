"""
This file acts on the extracted time series coordinates from the video using mediapipes and extracts features from them using tsfresh.
"""

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

data_path = Path(const["DATA_PATH"])
features_extracted_path = Path(const["FEATURES_EXTRACTED"])

# Define empty lists
list_0 = []
list_1 = []
y = []

# Read npy files and append to lists
for file in (data_path / features_extracted_path).glob('0/*.npy'):
    data = np.load(file)
    data_df = pd.DataFrame(data)
    list_0.append(data_df)
    y.append(0)
        
for file in (data_path / features_extracted_path).glob('1/*.npy'):
    data = np.load(file)
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

# Create output directory
output_path = data_path / const["TIMESERIES_FEATURES_EXTRACTED"]
output_path.mkdir(parents=True, exist_ok=True)

# Save X and y as pickle files
X.to_pickle(output_path / "X.pkl")
y.to_pickle(output_path / "y.pkl")
