import joblib
import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
import json
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class Model:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = joblib.load('./models/model.pkl')

        with open("./src/const.json", "r", encoding="utf-8") as f:
            const = json.load(f)
        
        # Read the settings from pkl file
        features_path = Path(const["DATA_PATH"]) / const["TIMESERIES_FEATURES_EXTRACTED"]
        with open(str(features_path / 'kind_to_fc_parameters.pkl'), "rb") as f:
            self.kind_to_fc_parameters = pickle.load(f)

    def predict(self, features):

        features = pd.DataFrame(features)
        features['id'] = 1
        X = extract_features(
            features,
            column_id='id',
            column_sort=None,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            impute_function=impute,
            n_jobs=1
            )
        
        proba = self.model.predict_proba(X)

        # Select based on threshold
        return (proba[:,1] >= self.threshold).astype(bool)

# model = Model()

# input = np.load('src/dataset_creation/6_features_extracted/1/vid_00001_00001.npy')

# # Perform the prediction multiple times and measure the execution time
# n_runs = 10
# execution_times = []
# for i in range(n_runs):
#     start_time = time.time()
#     output = model.predict(input)
#     end_time = time.time()
#     execution_times.append(end_time - start_time)
#     print(output)

# # Print the mean and standard deviation of the execution time
# print("Mean execution time: ", np.mean(execution_times), "seconds")
# print("Standard deviation of execution time: ", np.std(execution_times), "seconds")