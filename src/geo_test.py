from collections import defaultdict
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
#import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import LeaveOneOut
import torch
import torch.nn as nn

from geo_dataset import Dataset


__all__ = ["Tester"]


def true_positive(y_true, y_pred) -> int:
    return ((y_true == 1) * (y_pred == 1)).sum()

def true_negative(y_true, y_pred) -> int:
    return ((y_true == 0) * (y_pred == 0)).sum()

def false_positive(y_true, y_pred) -> int:
    return ((y_true == 0) * (y_pred == 1)).sum()

def false_negative(y_true, y_pred) -> int:
    return ((y_true == 1) * (y_pred == 0)).sum()


class Tester:

    def __init__(self, device: str = "cuda", verbose: bool = True):
        self._torch_device = device
        self.verbose = verbose

        self.SCORING = [
            "neg_log_loss", "f1", "precision", "recall", "roc_auc", "accuracy",
            "true_positive", "true_negative", "false_positive", "false_negative",
        ]
        self.METRICS = {
            "neg_log_loss": log_loss, "f1": f1_score, "precision": precision_score,
            "recall": recall_score, "roc_auc": roc_auc_score, "accuracy": accuracy_score,
            "true_positive": true_positive, "true_negative": true_negative,
            "false_positive": false_positive, "false_negative": false_negative,
        }
        self._LOGREG_ARGS = {"C": 100, "max_iter": 10000, "class_weight": "balanced", "n_jobs": -2}
        self._logreg_args = {**self._LOGREG_ARGS}

    def test(self, model: nn.Module, dataset: Dataset, logreg_args: dict = {}):
        # get testing datasets
        testing_df = dataset._data_test
        groups = testing_df.groupby("name")
        training_columns = dataset._training_columns

        # set logistic regression parameters
        self._logreg_args = {**self._LOGREG_ARGS, **logreg_args}

        model.eval()
        model.to(self._torch_device)

        log_metrics = defaultdict(list)

        for name, group in groups:
            group = group.reset_index(drop=True)
            tmp = pd.read_csv(dataset._path_prefix+group["file_location"][0])

            # raw input data
            x = torch.tensor(tmp[training_columns].to_numpy().astype("float32"), device=self._torch_device)
            encoded = model.encoder(x).detach().cpu().numpy()
            x = x.cpu().numpy().astype(np.float64)
            combined = np.concatenate([x, encoded], axis=1, dtype=np.float64)

            for _, row in group.iterrows():
                class_mapping = row["class_mapping"]
                y = tmp[row["target_column"]].astype("str").apply(lambda x: class_mapping[x]).to_numpy("float64")

                # Encoded
                start = perf_counter()
                y_pred, y_pred_proba = self._loocv(encoded, y)
                end = perf_counter()
                log_metrics["name"].append(name)
                log_metrics["target"].append(row["target_column"])
                log_metrics["input"].append("encoded")
                self._loocv_score(y, y_pred, y_pred_proba, log_metrics)
                log_metrics["time"].append(end-start)

                # Combined
                start = perf_counter()
                y_pred, y_pred_proba = self._loocv(combined, y)
                end = perf_counter()
                log_metrics["name"].append(name)
                log_metrics["target"].append(row["target_column"])
                log_metrics["input"].append("combined")
                self._loocv_score(y, y_pred, y_pred_proba, log_metrics)
                log_metrics["time"].append(end-start)

        df = pd.DataFrame.from_dict(log_metrics)
        return df

    def _loocv(self, X, y):
        loo = LeaveOneOut()

        y_pred = np.zeros_like(y)
        y_pred_proba = np.zeros_like(y)

        #shap_values = np.zeros_like(X)

        # LOOCV loop
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            class_ratio = y_train.mean()
            if 0 < class_ratio < 1:
                logreg = LogisticRegression(**self._logreg_args)
                logreg.fit(X_train, y_train)
                y_pred[test_idx] = logreg.predict(X_test)
                y_pred_proba[test_idx] = logreg.predict_proba(X_test)[0][1]
                #explainer = shap.LinearExplainer(logreg, X_train)
                #shap_values[test_idx, :] = explainer(X_test).values
            else: # if we only have 1 class in train data
                y_pred[test_idx] = round(class_ratio)
                y_pred_proba[test_idx] = class_ratio
                #shap_values[test_idx, :] = 0

        return y_pred, y_pred_proba

    def _loocv_score(self, y_true, y_pred, y_pred_proba, save_dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sname in self.SCORING:
                if sname == "neg_log_loss":
                    save_dict[sname].append(self.METRICS[sname](y_true, y_pred_proba))
                else:
                    save_dict[sname].append(self.METRICS[sname](y_true, y_pred))
