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

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True,
                    help='Folder containing the extracted features.')
args = parser.parse_args()

path_data = Path(args.folder)

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
    'random_state': [constants.SEED]
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
    'random_state': [constants.SEED]
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
    'random_state': [constants.SEED],
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

print("----------------------------------------")
print("KNN")

param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
knn = KNeighborsClassifier()

grid_search = build_grid_search(knn, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters KNN:", grid_search.best_params_)

training_results["KNN"] = {}
training_results["KNN"]["y_proba_test"] = grid_search.best_estimator_.predict_proba(
    X_test
)
training_results["KNN"]["y_proba_train"] = grid_search.best_estimator_.predict_proba(
    X_train
)
training_results["KNN"]["config"] = grid_search.best_estimator_.get_params()
training_results["KNN"]["model"] = grid_search

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("MLPClassifier")

param_grid = {
    "hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100)],
    "activation": ["relu"],
    "solver": ["adam"],
    "alpha": [0.0001],
    "batch_size": ["auto"],
    "learning_rate": ["constant"],
    "learning_rate_init": [0.001],
    "max_iter": [200],
    "random_state": [constants.SEED],
}
mlp = MLPClassifier()

grid_search = build_grid_search(mlp, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters MLP:", grid_search.best_params_)

training_results["MLP"] = {}
training_results["MLP"]["y_proba_test"] = grid_search.best_estimator_.predict_proba(
    X_test
)
training_results["MLP"]["y_proba_train"] = grid_search.best_estimator_.predict_proba(
    X_train
)
training_results["MLP"]["config"] = grid_search.best_estimator_.get_params()
training_results["MLP"]["model"] = grid_search

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("AdaBoostClassifier")

param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 1],
    "random_state": [constants.SEED],
}
ada = AdaBoostClassifier()

grid_search = build_grid_search(ada, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters Ada:", grid_search.best_params_)

training_results["Ada"] = {}
training_results["Ada"]["y_proba_test"] = grid_search.best_estimator_.predict_proba(
    X_test
)
training_results["Ada"]["y_proba_train"] = grid_search.best_estimator_.predict_proba(
    X_train
)
training_results["Ada"]["config"] = grid_search.best_estimator_.get_params()
training_results["Ada"]["model"] = grid_search

# -----------------------------------------------------------------------------

print("----------------------------------------")
print("GaussianNB")

param_grid = {
    "var_smoothing": [1e-09],
}

gnb = GaussianNB()

grid_search = build_grid_search(gnb, param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters GNB:", grid_search.best_params_)

training_results["GNB"] = {}
training_results["GNB"]["y_proba_test"] = grid_search.best_estimator_.predict_proba(
    X_test
)
training_results["GNB"]["y_proba_train"] = grid_search.best_estimator_.predict_proba(
    X_train
)
training_results["GNB"]["config"] = grid_search.best_estimator_.get_params()
training_results["GNB"]["model"] = grid_search

# -----------------------------------------------------------------------------

# Save training_results
with open(str(path_data / 'training_results.pkl'), 'wb') as handle:
    pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------

# Evaluate performance on wandb

# Iterate on training_results
for classifier_name, model_details in training_results.items():

    # start a new wandb run and add your model hyperparameters
    wandb.init(
        project=constants.MODEL_NAME,
        config=model_details.get('config'),
        name=classifier_name+"_ext_neg_teo",
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