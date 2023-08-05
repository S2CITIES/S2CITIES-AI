# Generate train, test split for the dataset
# but not by splitting all the loaded data, but by
# generating annotations in a txt format assigning each
# video either to train or test.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import constants


if __name__ == '__main__':

     # Read arguments with argparse: test size and shuffle
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size for train_test_split.')
    parser.add_argument('--folder', type=str, required=True,
                        help='Folder containing the extracted features.')
    args = parser.parse_args()

    path_data = Path(args.folder)

    if not Path(path_data / 'X.pkl').is_file() or not Path(path_data / 'y.pkl').is_file():
        raise FileNotFoundError('X.pkl or y.pkl not found in {}'.format(path_data))

    # Read data
    X = pd.read_pickle(path_data / 'X.pkl')
    y = pd.read_pickle(path_data / 'y.pkl')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=constants.SEED,
        shuffle=True,
        stratify=y
    )

    # Write them to the same folder
    X_train.to_pickle(path_data / 'X_train.pkl')
    X_test.to_pickle(path_data / 'X_test.pkl')
    y_train.to_pickle(path_data / 'y_train.pkl')
    y_test.to_pickle(path_data / 'y_test.pkl')

    print(f'Train and test sets saved in {path_data}')
