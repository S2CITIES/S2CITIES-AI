"""
This class evaluates a generic binary classification model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

class ModelEvaluator:

    def __init__(self, model_name, y_true, y_proba, threshold=0.5):
        self.model_name = model_name
        self.y_true = y_true
        self.y_proba = y_proba
        self.threshold = threshold
        self.y_pred = np.where(y_proba > self.threshold, 1, 0)

    def evaluate(self):

        # Calculate accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print("Accuracy: ", accuracy)

        # Calculate AUC
        auc = roc_auc_score(self.y_true, self.y_proba)
        print("AUC: ", auc)

        # Calculate confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(self.y_true, self.y_pred)
        ax.set_title(f"{self.model_name} Confusion Matrix")
        plot_confusion_matrix(conf_mat=cm,
                              show_absolute=True,
                              show_normed=True,
                              colorbar=True, figure=fig, axis=ax)

        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.show()
