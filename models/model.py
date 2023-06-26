import joblib
import pandas as pd
import numpy as np
import time
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class Model:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = joblib.load('./models/random_forest.pkl')



    def predict(self, features):

        features = pd.DataFrame(features)
        features['id'] = 1
        X = extract_features(features, column_id='id', column_sort=None,
                     # default_fc_parameters=extraction_settings,
                     # we impute = remove all NaN features automatically
                     impute_function=impute,
                     n_jobs=1)
        
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