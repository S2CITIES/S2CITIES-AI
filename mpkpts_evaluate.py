import argparse
import pickle
from pathlib import Path

import pandas as pd

from src import constants
from src.modelevaluator import ModelEvaluator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder",
    type=str,
    required=True,
    help="Folder containing the extracted features.",
)
args = parser.parse_args()

path_data = Path(args.folder)

y_test = pd.read_pickle(path_data / "y_test.pkl")
with open(str(path_data / "training_results.pkl"), "rb") as handle:
    training_results = pickle.load(handle)

y_proba_dict = {
    classifier_name: model_details.get("y_proba_test")[:, 1]
    for classifier_name, model_details in training_results.items()
}

# Create "report" folder if it doesn't exist
report_folder = Path("report") / constants.MODEL_NAME
report_folder.mkdir(parents=True, exist_ok=True)

# Evaluate performance on testing data
evaluator = ModelEvaluator(y_true=y_test, y_proba_dict=y_proba_dict, threshold=0.5)
evaluator.get_metrics(export="all", filename=f"./report/{constants.MODEL_NAME}/stats")
evaluator.plot_roc_curve(export="save", filename=f"./report/{constants.MODEL_NAME}/roc")
evaluator.plot_precision_recall_curve(
    export="save", filename=f"./report/{constants.MODEL_NAME}/precision_recall"
)
evaluator.plot_confusion_matrix(
    export="save", filename=f"./report/{constants.MODEL_NAME}/confusion_matrix"
)

# TODO compute prediction time (with confidence interval)
