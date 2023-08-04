"""
This file trains a model on the extracted features using statistical methods.
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import pickle
from pathlib import Path
from src import constants

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc


def build_grid_search(estimator, param_grid):
    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=param_grid,
                               cv=StratifiedKFold(),
                               n_jobs=-1,
                               verbose=2,
                               scoring='accuracy')
    return grid_search

path_data = path_data = Path(constants.TIMESERIES_FEATURES_EXTRACTED)

# Read data
X_train = pd.read_pickle(path_data / 'X_train.pkl')
X_test = pd.read_pickle(path_data / 'X_test.pkl')
y_train = pd.read_pickle(path_data / 'y_train.pkl')
y_test = pd.read_pickle(path_data / 'y_test.pkl')

# Read final features
with open(str(path_data / 'final_features.pkl'), 'rb') as handle:
    final_features = pickle.load(handle)

# Subset data to final features
X_train = X_train[final_features]
X_test = X_test[final_features]

# Normalize data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Save scaler
with open(str(path_data / 'scaler.pkl'), 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

training_results = {}

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("Random Forest")

param_grid = {
    'n_estimators': [150, 300, 450],
    'max_depth': [None, 8, 16, 24],
    'min_samples_split': [1, 8, 16, 24],
    'min_samples_leaf': [1, 8, 16, 24],
    'random_state': [42]
}
rf = RandomForestClassifier()

grid_search = build_grid_search(rf, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

training_results['RF'] = {}
training_results['RF']['y_proba_test'] = grid_search.best_estimator_.predict_proba(X_test)
training_results['RF']['y_proba_train'] = grid_search.best_estimator_.predict_proba(X_train)
training_results['RF']['config'] = grid_search.best_estimator_.get_params()
training_results['RF']['model'] = grid_search

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("SVM")

param_grid = {
    'C': [1],
    'gamma': ['scale'],
    'kernel': ['rbf'],
    'random_state': [42]
}
svc = SVC(probability=True)

grid_search = build_grid_search(svc, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters SVM:", grid_search.best_params_)

training_results['SVM'] = {}
training_results['SVM']['y_proba_test'] = grid_search.best_estimator_.predict_proba(X_test)
training_results['SVM']['y_proba_train'] = grid_search.best_estimator_.predict_proba(X_train)
training_results['SVM']['config'] = grid_search.best_estimator_.get_params()
training_results['SVM']['model'] = grid_search

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("Logistic Regression")

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000],
    'random_state': [42],
}
lr = LogisticRegression()

grid_search = build_grid_search(lr, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters LR:", grid_search.best_params_)

training_results['LR'] = {}
training_results['LR']['y_proba_test'] = grid_search.best_estimator_.predict_proba(X_test)
training_results['LR']['y_proba_train'] = grid_search.best_estimator_.predict_proba(X_train)
training_results['LR']['config'] = grid_search.best_estimator_.get_params()
training_results['LR']['model'] = grid_search

# -----------------------------------------------------------------------------

# Save training_results
with open(str(path_data / 'training_results.pkl'), 'wb') as handle:
    pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------

# Evaluate performance on wandb

# Iterate on training_results
for model_name, model_details in training_results.items():

    # start a new wandb run and add your model hyperparameters
    wandb.init(
        project=constants.MODEL_NAME,
        config=model_details.get('config'),
        name=model_name,
        )

    # log additional visualisations to wandb
    plot_class_proportions(y_train, y_test, constants.LABELS)
    #plot_learning_curve(model_details.get('model'), X_train, y_train)
    plot_roc(y_test, model_details.get('y_proba_test'), constants.LABELS)
    plot_precision_recall(y_test, model_details.get('y_proba_test'), constants.LABELS)
    
    # log metrics to wandb
    y_pred_test = np.where(model_details.get('y_proba_test')[:, 1] > 0.5, 1, 0)
    y_pred_train = np.where(model_details.get('y_proba_train')[:, 1] > 0.5, 1, 0)
    wandb.log({
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_auc": roc_auc_score(y_test, model_details.get('y_proba_test')[:, 1]),
        "test_precision": precision_score(y_test, y_pred_test),
        "test_recall": recall_score(y_test, y_pred_test),
        "test_f1": f1_score(y_test, y_pred_test),
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_auc": roc_auc_score(y_train, model_details.get('y_proba_train')[:, 1]),
        "train_precision": precision_score(y_train, y_pred_train),
        "train_recall": recall_score(y_train, y_pred_train),
        "train_f1": f1_score(y_train, y_pred_train),
        })

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()