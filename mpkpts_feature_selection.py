from src import constants
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
import pickle
import os
import argparse

if __name__ == '__main__':

    # Read arguments with argparse: n_jobs
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of jobs for parallelization.')
    parser.add_argument('--folder', type=str, required=True,
                        help='Folder containing the extracted features.')
    args = parser.parse_args()

    # Define seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(constants.SEED)
    np.random.seed(constants.SEED)

    path_data = Path(args.folder)

    # Read data
    X = pd.read_pickle(path_data / 'X.pkl')
    y = pd.read_pickle(path_data / 'y.pkl')

    class_labels = {
        0: 'SFH_No',
        1: 'SFH_Yes',
    }

    # TODO decide to keep this or not
    # Subset X and y to have balanced classes
    # minority_class = y.value_counts().min()
    # minority_class_value = y.value_counts().idxmin()
    # X = X[y == minority_class_value].append(X[y == (1-minority_class_value)].sample(n=minority_class, random_state=constants.SEED))
    # y = y[y == minority_class_value].append(y[y == (1-minority_class_value)].sample(n=minority_class, random_state=constants.SEED))

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y,
        )
    class_weights = dict(zip(np.unique(y), class_weights))

    # Normalize data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Instantiate feature selector
    rf_selector = SelectFromModel(
        RandomForestClassifier(
            criterion='entropy',
            max_depth=10,
            n_estimators=250,
            random_state=constants.SEED,
            class_weight=class_weights
        ),
        max_features=30
    )

    # Fit feature selector
    print('Fitting RF selector...')
    rf_selector.fit(X, y)

    # Get selected features
    selected_feature_indices_rf = rf_selector.get_support(indices=True)
    selected_feature_names_rf = [X.columns[index] for index in selected_feature_indices_rf]

    # Print selected features
    # print(f'Selected {len(selected_feature_names_rf)} features:')
    # print(selected_feature_names_rf)

    selected_features = selected_feature_names_rf

    # Create a classifier with somehow good parameters
    # to perform stepwise feature selection
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=constants.SEED,
    )

    # Backward selection
    print('Performing backward selection...')
    sfs_backward = SequentialFeatureSelector(
        classifier,
        k_features=1,
        forward=False,
        floating=False,
        scoring='roc_auc',
        verbose=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=constants.SEED),
        n_jobs=args.n_jobs,
        ).fit(X[selected_features], y)

    # Save the results
    with open(str(path_data / 'sfs_backward.pkl'), 'wb') as handle:
        pickle.dump(sfs_backward, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Forward selection
    print('Performing forward selection...')
    sfs_forward = SequentialFeatureSelector(
        classifier,
        k_features=30,
        forward=True,
        floating=False,
        scoring='roc_auc',
        verbose=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=constants.SEED),
        n_jobs=args.n_jobs,
        ).fit(X[selected_features], y)

    # Save the results
    with open(str(path_data / 'sfs_forward.pkl'), 'wb') as handle:
        pickle.dump(sfs_forward, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

