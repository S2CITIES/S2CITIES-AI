"""
This file trains a model on the extracted features using statistical methods.
"""

import os
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

from modelevaluator import ModelEvaluator

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read setting the first column as index
X = pd.read_csv('dataset_creation/6_features_filtered.csv', index_col=0)
#X = pd.read_pickle('dataset_creation/6_X.pkl')

# Read y from pickle
y = pd.read_pickle('dataset_creation/6_y.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# Normalize data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train logistic regression model on training data with strong L1 regularization
# to select most important features
lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
lr.fit(X_train, y_train)
selector = SelectFromModel(lr, prefit=True)
X_train_selected = pd.DataFrame(selector.transform(X_train))
X_train_selected.columns = X_train.columns[selector.get_support()]

# Show selected features names and length
print("Selected features:", X_train.columns[selector.get_support()])
print("Number of selected features:", len(X_train.columns[selector.get_support()]))

# Train a random forest classifier on the selected features
# and tune hyperparameters using grid search
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_split': [2, 5, 10, 15, 20],
#     'min_samples_leaf': [1, 2, 5, 10, 15, 20],
#     'random_state': [42]
# }
# rf = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=param_grid,
#                            cv=StratifiedKFold(),
#                            n_jobs=-1,
#                            verbose=2,
#                            scoring='accuracy')
# grid_search.fit(X_train_selected, y_train)

# # Show best parameters
# print("Best parameters RF:", grid_search.best_params_)

# Tune a svm classifier on the selected features
# and tune hyperparameters using grid search
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
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


# Evaluate performance on testing data
X_test_selected = pd.DataFrame(selector.transform(X_test))
X_test_selected.columns = X_test.columns[selector.get_support()]
y_test_proba = grid_search.best_estimator_.predict_proba(X_test_selected)[:,1]

evaluator = ModelEvaluator(model_name='SVM', y_true=y_test, y_proba=y_test_proba, threshold=0.5)
evaluator.evaluate_metrics()
evaluator.plot_roc_curve()
evaluator.plot_confusion_matrix()
