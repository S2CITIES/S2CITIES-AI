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

# Read X from pickle
X = pd.read_pickle('dataset_creation/6_X.pkl')

# Read y from pickle
y = pd.read_pickle('dataset_creation/6_y.pkl')

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectFromModel(
        estimator=LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ))
])

# Fit the pipeline to the data
pipeline.fit(X, y)

# Save the pipeline to a pickle
joblib.dump(pipeline, 'model.pkl')

sys.exit(0)
# -----------------------------------------------------------------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
    )

# Normalize data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train logistic regression model on training data with strong L1 regularization
# to select most important features
selector = SelectFromModel(
    LogisticRegression(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        #class_weight=class_weights
    )
).fit(X_train, y_train)

X_train_selected = pd.DataFrame(selector.transform(X_train))
X_train_selected.columns = X_train.columns[selector.get_support()]

X_test_selected = pd.DataFrame(selector.transform(X_test))
X_test_selected.columns = X_test.columns[selector.get_support()]

# Show selected features names and length
print("Selected features:", X_train.columns[selector.get_support()])
print("Number of selected features:", len(X_train.columns[selector.get_support()]))

y_proba_dict = {}

# -----------------------------------------------------------------------------

# Train a random forest classifier on the selected features
# and tune hyperparameters using grid search
param_grid = {
    #'n_estimators': [100, 200, 300, 400, 500],
    'n_estimators': [400],
    #'max_depth': [None, 5, 10, 15, 20],
    'max_depth': [None],
    #'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_split': [5],
    #'min_samples_leaf': [1, 2, 5, 10, 15, 20],
    'min_samples_leaf': [2],
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
    'kernel': ['sigmoid'],
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
    'C': [10],
    #'penalty': ['l1', 'l2'],
    'penalty': ['l2'],
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
evaluator.get_metrics(export="all", filename="../report/stats")
evaluator.plot_roc_curve(export="save", filename='../report/roc')
evaluator.plot_precision_recall_curve(export="save", filename='../report/precision_recall')
evaluator.plot_confusion_matrix(export="save", filename='../report/confusion_matrix')

# lavorato fin'ora coi video
# 1: da vid_00097_00022 a vid_00161_00105
# 0: da vid_00097_00025 a vid_00161_00107
