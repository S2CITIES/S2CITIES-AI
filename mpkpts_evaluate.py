from src.modelevaluator import ModelEvaluator
from src import constants
from pathlib import Path
import pickle
import pandas as pd

path_data = path_data = Path(constants.TIMESERIES_FEATURES_EXTRACTED)

y_test = pd.read_pickle(path_data / 'y_test.pkl')
with open(str(path_data / 'training_results.pkl'), 'rb') as handle:
    training_results = pickle.load(handle)


y_proba_dict = {
    model_name: model_details.get('y_proba_test')[:, 1]
    for model_name, model_details in training_results.items()
    }


# Evaluate performance on testing data
evaluator = ModelEvaluator(y_true=y_test, y_proba_dict=y_proba_dict, threshold=0.5)
evaluator.get_metrics(export="all", filename=f"./report/{constants.MODEL_NAME}/stats")
evaluator.plot_roc_curve(export="save", filename=f'./report/{constants.MODEL_NAME}/roc')
evaluator.plot_precision_recall_curve(export="save", filename=f'./report/{constants.MODEL_NAME}/precision_recall')
evaluator.plot_confusion_matrix(export="save", filename=f'./report/{constants.MODEL_NAME}/confusion_matrix')

