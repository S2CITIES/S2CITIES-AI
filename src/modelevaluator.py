"""
This class evaluates a generic binary classification model.
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def exportable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        export = kwargs.pop("export", None)
        filename = kwargs.pop("filename", "plot")
        result = func(*args, **kwargs)
        if export is not None and export not in ["save", "show", "both"]:
            raise ValueError("Invalid export option")
        if export in ["save", "both"]:
            plt.savefig(f"{filename}.pdf", bbox_inches="tight")
        if export in ["show", "both"]:
            plt.show()
        return result

    return wrapper


def exportable_dataframe(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        export = kwargs.pop("export", None)
        filename = kwargs.pop("filename", "data")
        result = func(*args, **kwargs)
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Result must be a pandas DataFrame")
        if export is not None and export not in ["csv", "latex", "print", "all"]:
            raise ValueError("Invalid export option")
        if export in ["csv", "all"]:
            csv_filename = f"{filename}.csv"
            result.to_csv(csv_filename, index=False)
        if export in ["latex", "all"]:
            latex_filename = f"{filename}.tex"
            with open(latex_filename, "w") as f:
                # .hide(axis="index") to hide the index after .style
                f.write(
                    result.set_index("Model")
                    .style.highlight_max(axis=0, props="textbf:--rwrap;")
                    .to_latex(hrules=True)
                )
        if export in ["print", "all"]:
            print(result)
        return result

    return wrapper


class ModelEvaluator:
    def __init__(self, y_true, y_proba_dict, threshold=0.5):
        """
        Initializes the ModelEvaluator class.

        Args:
            y_true (array-like): True labels.
            y_proba_dict (dict): Dictionary of model names and their predicted probabilities.
            threshold (float): Threshold for binary classification. Defaults to 0.5.
        """
        self.y_true = y_true
        self.y_proba_dict = y_proba_dict
        self.threshold = threshold
        self.y_pred_dict = {
            classifier_name: np.where(y_proba > self.threshold, 1, 0)
            for classifier_name, y_proba in self.y_proba_dict.items()
        }
        self.metrics_df = pd.DataFrame(
            columns=[
                "Model",
                "Accuracy",
                "AUC",
                "Precision",
                "Recall",
                "F1 score",
            ]
        )
        for classifier_name, y_pred in self.y_pred_dict.items():
            # Calculate accuracy
            accuracy = accuracy_score(self.y_true, y_pred)

            # Calculate AUC
            auc = roc_auc_score(self.y_true, self.y_proba_dict[classifier_name])

            # Calculate precision
            precision = precision_score(self.y_true, y_pred)

            # Calculate recall
            recall = recall_score(self.y_true, y_pred)

            # Calculate F1 score
            f1 = f1_score(self.y_true, y_pred)

            # Add metrics to dataframe
            self.metrics_df = pd.concat(
                [
                    self.metrics_df,
                    pd.DataFrame(
                        {
                            "Model": [classifier_name],
                            "Accuracy": [accuracy],
                            "AUC": [auc],
                            "Precision": [precision],
                            "Recall": [recall],
                            "F1 score": [f1],
                        }
                    ),
                ],
                ignore_index=True,
            )

    @exportable_dataframe
    def get_metrics(self):
        """
        Returns a pandas DataFrame with evaluation metrics for each model.

        Args:
            export (str): Export option. Can be "csv", "latex", "print", or "all".
            filename (str): Filename for exported file.
        """
        return self.metrics_df

    @exportable
    def plot_roc_curve(self):
        """
        Plots the ROC curve for each model.

        Args:
            export (str): Export option. Can be "save", "show", or "both".
            filename (str): Filename for exported file.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for classifier_name, y_proba in self.y_proba_dict.items():
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(self.y_true, y_proba)

            # Calculate AUC
            auc = roc_auc_score(self.y_true, y_proba)

            # Plot ROC curve
            ax.plot(
                fpr,
                tpr,
                label=f"{classifier_name} (AUC = {auc:.4f})",
            )

        ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

    def plot_confusion_matrix(self, export, filename):
        """
        Plots the confusion matrix for each model.

        Args:
            export (str): Export option. Can be "save", "show", or "both".
            filename (str): Filename for exported file.
        """

        for classifier_name, y_pred in self.y_pred_dict.items():
            fig, ax = plt.subplots(figsize=(10, 10))
            cm = confusion_matrix(self.y_true, y_pred)
            ax.set_title(f"{classifier_name} Confusion Matrix")
            plot_confusion_matrix(
                conf_mat=cm,
                show_absolute=True,
                show_normed=True,
                colorbar=True,
                figure=fig,
                axis=ax,
            )

            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")

            if export is not None and export not in ["save", "show", "both"]:
                raise ValueError("Invalid export option")
            if export in ["save", "both"]:
                plt.savefig(f"{filename}_{classifier_name}.pdf", bbox_inches="tight")
            if export in ["show", "both"]:
                plt.show()

    @exportable
    def plot_precision_recall_curve(self):
        """
        Plots the precision-recall curve for each model.

        Args:
            export (str): Export option. Can be "save", "show", or "both".
            filename (str): Filename for exported file.
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        for classifier_name, y_proba in self.y_proba_dict.items():
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(self.y_true, y_proba)

            # Plot precision-recall curve
            ax.plot(
                recall,
                precision,
                label=f"{classifier_name}",
            )

        # Compute the zero skill model line
        # It will depend on the fraction of observations belonging to the positive class
        zero_skill = len(self.y_true[self.y_true == 1]) / len(self.y_true)

        # Compute the perfect model line
        perfect_precision = np.ones_like(self.y_true)
        perfect_recall = np.linspace(0, 1, num=len(perfect_precision))

        # Plot zero skill and perfect model lines
        plt.plot([0, 1], [zero_skill, zero_skill], "b--", label="Zero skill")
        plt.plot(
            perfect_recall, perfect_precision, "g--", linewidth=2, label="Perfect model"
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower right")
