import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


class Model:
    def __init__(self,
                 training_results,
                 model_choice,
                 tsfresh_parameters,
                 scaler,
                 final_features,
                 ):
        
        with open(training_results, 'rb') as handle:
            self.training_results = pickle.load(handle)

        self.model = self.training_results[model_choice].get('model')

        with open(tsfresh_parameters, "rb") as f:
            self.kind_to_fc_parameters = pickle.load(f)

        with open(scaler, 'rb') as handle:
            self.scaler = pickle.load(handle)

        self.final_features = pd.read_pickle(final_features)

    def predict(self, features):

        features = pd.DataFrame(features)
        features['id'] = 1
        # save features
        features.to_pickle('features.pkl')
        X = extract_features(
            features,
            column_id='id',
            column_sort=None,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            impute_function=impute,
            n_jobs=1
            )

        # Reorder columns to match the training set        
        X = X[self.final_features]

        # Scale the features
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        # Predict the output
        proba = self.model.predict_proba(X)

        # Select based on threshold
        return proba

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