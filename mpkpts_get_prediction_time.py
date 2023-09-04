import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from src import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_results",
        type=str,
        required=True,
        help="Path to the training results file",
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing the extracted features.",
    )

    args = parser.parse_args()

    path_data = Path(args.folder)

    # Read data
    X_train = pd.read_pickle(path_data / "X_train.pkl")
    X_test = pd.read_pickle(path_data / "X_test.pkl")
    y_train = pd.read_pickle(path_data / "y_train.pkl")
    y_test = pd.read_pickle(path_data / "y_test.pkl")

    # Read final features
    with open(str(path_data / "final_features.pkl"), "rb") as handle:
        final_features = pickle.load(handle)

    # Subset data to final features
    X_train = X_train[final_features]
    X_test = X_test[final_features]

    # Normalize data
    with open(str(path_data / "scaler.pkl"), "rb") as handle:
        scaler = pickle.load(handle)

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Merge X train and X test
    X = pd.concat([X_train, X_test], axis=0)

    # Load training results
    with open(args.training_results, "rb") as handle:
        training_results = pickle.load(handle)

    # Create result list (to be converted to dataframe)
    result_df = []

    # Compute prediction time for each model for each sample
    for model_name, model_details in training_results.items():
        # Load the model
        model = model_details.get("model")

        times = []

        # Compute prediction time for each sample
        for i in range(X.shape[0]):
            sample = X.iloc[[i]]
            start_time = time.time()
            model.predict(sample)
            end_time = time.time()
            prediction_time = end_time - start_time

            times.append(prediction_time)

        # Compute mean and 95% confidence interval
        alpha = 0.05
        mean_prediction_time = np.mean(times)
        std_prediction_time = np.std(times)

        # Add to dataframe
        result_df.append(
            {
                "Model": model_name,
                "mean": mean_prediction_time,
                "lower_bound_95": mean_prediction_time
                - st.t.ppf(1 - alpha / 2, len(times) - 1) * std_prediction_time,
                "upper_bound_95": mean_prediction_time
                + st.t.ppf(1 - alpha / 2, len(times) - 1) * std_prediction_time,
            }
        )

    # Convert to dataframe
    result_df = pd.DataFrame(result_df)

    # max between value and 0 to avoid negative values
    result_df["lower_bound_95"] = result_df["lower_bound_95"].apply(
        lambda x: np.max([0, x])
    )

    # Sort by mean prediction time
    result_df = result_df.sort_values(by="mean", ascending=True)

    # Export to csv and tex
    report_folder = Path("report") / constants.MODEL_NAME
    result_df.to_csv(str(report_folder / "prediction_time.csv"), index=False)

    latex_filename = str(report_folder / "prediction_time.tex")
    with open(latex_filename, "w") as f:
        # .hide(axis="index") to hide the index after .style
        f.write(
            result_df.set_index("Model")
            .style.highlight_min(axis=0, props="textbf:--rwrap;")
            .to_latex(hrules=True)
        )
