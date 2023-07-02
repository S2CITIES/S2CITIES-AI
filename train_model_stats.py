"""
This file trains a model on the extracted features using statistical methods.
"""

import json
import multiprocessing
import os
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tsfresh
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight

from src.modelevaluator import ModelEvaluator

# Print how many CPUs are available
print(f'Number of CPUs: {multiprocessing.cpu_count()}')

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# Define paths
data_path = Path(const["DATA_PATH"])
features_path = data_path / const["TIMESERIES_FEATURES_EXTRACTED"]
report_path = Path('report')

# Define seed for reproducibility
SEED = 42
#random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
#tf.random.set_seed(SEED)
#tf.compat.v1.set_random_seed(SEED)


# Read data
X = pd.read_pickle(features_path / 'X.pkl')
y = pd.read_pickle(features_path / 'y.pkl')
class_labels = {
    0: 'SFH_No',
    1: 'SFH_Yes',
}

# Exploration
print(X.shape)
print(y.shape)

# Print how many samples of each class
print(y.value_counts())

# Store the minority class numerosity and value
minority_class = y.value_counts().min()
minority_class_value = y.value_counts().idxmin()

# Subset X and y to have balanced classes
# TODO decide to keep this or not
X = X[y == minority_class_value].append(X[y == (1-minority_class_value)].sample(n=minority_class, random_state=42))
y = y[y == minority_class_value].append(y[y == (1-minority_class_value)].sample(n=minority_class, random_state=42))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)


# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
print(class_weights)


# Normalize data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)




# Feature selection

# # Select with logistic (too slow)
# logistic_selector = SelectFromModel(
#     LogisticRegression(
#         C=0.05,
#         penalty='l1',
#         solver='liblinear',
#         max_iter=1000,
#         random_state=SEED,
#         class_weight=class_weights,
#         verbose=1
#     )
# )

# logistic_selector.fit(X_train, y_train)
# selected_feature_indices_lr = logistic_selector.get_support(indices=True)
# selected_feature_names_lr = [X_train.columns[index] for index in selected_feature_indices_lr]

# print(f'Selected {len(selected_feature_names_lr)} features:')
# selected_feature_names_lr

rf_selector = SelectFromModel(
    RandomForestClassifier(
        criterion='entropy',
        max_depth=10,
        n_estimators=250,
        random_state=SEED,
        class_weight=class_weights
    ),
    max_features=150
)

rf_selector.fit(X_train, y_train)
selected_feature_indices_rf = rf_selector.get_support(indices=True)
selected_feature_names_rf = [X_train.columns[index] for index in selected_feature_indices_rf]

print(f'Selected {len(selected_feature_names_rf)} features:')
print(selected_feature_names_rf)

#selected_features = list(set(selected_feature_names_rf+selected_feature_names_lr))
selected_features = selected_feature_names_rf

classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)

# Backward selection
sfs_backward = SequentialFeatureSelector(
    classifier,
    k_features=1,
    forward=False,
    floating=False,
    scoring='roc_auc',
    verbose=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=8,
    ).fit(X_train[selected_features], y_train)

# Save the results
# with open(str(data_path / 'sfs_backward.pkl'), 'wb') as handle:
#     pickle.dump(sfs_backward, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the results
with open(str(data_path / 'sfs_backward.pkl'), 'rb') as handle:
    sfs_backward = pickle.load(handle)

# Plot the results
plot_sfs(sfs_backward.get_metric_dict(), kind='std_dev',figsize=(10, 8))
plt.title('Sequential Backward Selection')
plt.xticks(np.arange(0, 155, 5))
plt.grid()
plt.savefig(str(report_path / 'feature_selection_backward.pdf'), bbox_inches='tight')
plt.show()

# Forward selection
sfs_forward = SequentialFeatureSelector(
    classifier,
    k_features=30,
    forward=True,
    floating=False,
    scoring='roc_auc',
    verbose=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    n_jobs=8,
    ).fit(X_train[selected_features], y_train)

# Save the results
# with open(str(data_path / 'sfs_forward.pkl'), 'wb') as handle:
#     pickle.dump(sfs_backward, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the results
with open(str(data_path / 'sfs_forward.pkl'), 'rb') as handle:
    sfs_forward = pickle.load(handle)

# Plot the results
plot_sfs(sfs_forward.get_metric_dict(), kind='std_dev',figsize=(10, 8))
plt.title('Sequential Forward Selection')
plt.xticks(np.arange(0, 155, 5))
plt.grid()
plt.savefig(str(report_path / 'feature_selection_forward.pdf'), bbox_inches='tight')
plt.show()




final_features = list(
    set(sfs_backward.subsets_[7]['feature_names']) |
    set(sfs_forward.subsets_[7]['feature_names'])
    )

print(f'Number of final features: {len(final_features)}')
print(final_features)



# Transform the data
X_train_selected = X_train[final_features]
X_test_selected = X_test[final_features]


# Save the extraction settings

# Construct the timeseries extraction settings
kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(
    X_test_selected
)

# Turn all keys to integers in kind_to_fc_parameters
kind_to_fc_parameters = {int(k): v for k, v in kind_to_fc_parameters.items()}

# Save the settings object
with open(str(features_path / 'kind_to_fc_parameters.pkl'), "wb") as f:
    pickle.dump(kind_to_fc_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)









# Define empty lists
list_0 = []

features_extracted_path = Path(const["FEATURES_EXTRACTED"])

# Read npy files and append to lists
for file in (data_path / features_extracted_path).glob("0/*.npy"):
    data = np.load(file)
    data_df = pd.DataFrame(data)
    list_0.append(data_df)
    break

# Convert lists to dataframes
id = 0
df_list = []
for i, data_list in enumerate([list_0]):
    for j, data_df in enumerate(data_list):
        data_df["id"] = id
        id += 1
        df_list.append(data_df)

df = pd.concat(df_list, ignore_index=True)

# Extract features using tsfresh
X_temp = extract_features(
    df,
    column_id="id",
    kind_to_fc_parameters=kind_to_fc_parameters,
    column_sort=None,
    impute_function=impute,  # we impute = remove all NaN features automatically
    n_jobs=0,  # Doesn't work with n_jobs=1,2,3,4 https://stackoverflow.com/a/62655746
    )

# X_temp.columns














# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('selector', SelectFromModel(
    #     estimator=LogisticRegression(
    #         C=0.1,
    #         penalty='l1',
    #         solver='liblinear',
    #         max_iter=1000,
    #         random_state=42
    #     )
    # )),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=15,
        min_samples_leaf=1,
        random_state=42
    ))
])

# Fit the pipeline to the data
pipeline.fit(X[X_temp.columns], y)

# Save the pipeline to a pickle
joblib.dump(pipeline, 'models/model.pkl')

sys.exit(0)

y_proba_dict = {}

# -----------------------------------------------------------------------------

# Train a random forest classifier on the selected features
# and tune hyperparameters using grid search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    #'n_estimators': [100],
    'max_depth': [None, 5, 10, 15, 20],
    #'max_depth': [10],
    'min_samples_split': [2, 5, 10, 15, 20],
    #'min_samples_split': [15],
    'min_samples_leaf': [1, 2, 5, 10, 15, 20],
    #'min_samples_leaf': [1],
    'random_state': [42]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=StratifiedKFold(),
                           n_jobs=-1,
                           verbose=2,
                           scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Show best parameters
print("Best parameters RF:", grid_search.best_params_)

y_proba_dict['RF'] = grid_search.best_estimator_.predict_proba(X_test_selected)[:,1]

# -----------------------------------------------------------------------------

# Tune a SVM classifier on the selected features
# and tune hyperparameters using grid search
param_grid = {
    #'C': [0.1, 1, 10, 100, 1000],
    'C': [1],
    #'gamma': ['scale', 'auto'],
    'gamma': ['scale'],
    #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'kernel': ['rbf'],
    'random_state': [42]
}
svc = SVC(probability=True)
grid_search = GridSearchCV(estimator=svc,
                           param_grid=param_grid,
                           cv=StratifiedKFold(),
                           n_jobs=-1,
                           verbose=2,
                           scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Show best parameters
print("Best parameters SVM:", grid_search.best_params_)

y_proba_dict['SVM'] = grid_search.best_estimator_.predict_proba(X_test_selected)[:,1]

# -----------------------------------------------------------------------------

# Train a logistic regression
# and tune hyperparameters using grid search
param_grid = {
    #'C': [0.1, 1, 10, 100, 1000],
    'C': [1],
    #'penalty': ['l1', 'l2'],
    'penalty': ['l1'],
    'solver': ['liblinear'],
    'max_iter': [1000],
    'random_state': [42],
}
lr = LogisticRegression()
grid_search = GridSearchCV(estimator=lr,
                           param_grid=param_grid,
                           cv=StratifiedKFold(),
                           n_jobs=-1,
                           verbose=2,
                           scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Show best parameters
print("Best parameters LR:", grid_search.best_params_)

y_proba_dict['LR'] = grid_search.best_estimator_.predict_proba(X_test_selected)[:,1]

# -----------------------------------------------------------------------------

# Evaluate performance on testing data
evaluator = ModelEvaluator(y_true=y_test, y_proba_dict=y_proba_dict, threshold=0.5)
evaluator.get_metrics(export="all", filename="./report/stats")
evaluator.plot_roc_curve(export="save", filename='./report/roc')
evaluator.plot_precision_recall_curve(export="save", filename='./report/precision_recall')
evaluator.plot_confusion_matrix(export="save", filename='./report/confusion_matrix')

# lavorato fin'ora coi video
# 1: da vid_00097_00022 a vid_00161_00105
# 0: da vid_00097_00025 a vid_00161_00107
