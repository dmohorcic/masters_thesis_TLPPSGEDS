from dataclasses import dataclass
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


__all__ = ["Dataset"]


class Dataset:

    @dataclass
    class TestInfo:
        n_columns: int = 0
        n_samples: int = 0
        n_tasks: int = 0
        _X: torch.Tensor = None
        _Y: torch.Tensor = None
        _W: torch.Tensor = None
        dataset_idx: list = None
        idx_to_dataset: dict = None
        target_val: list = None

    def __init__(self, info_path: str = "data/GEO_v2/training_data_v2.csv",
                 column_path: str = "data/GEO_v2/training_columns.txt",
                 train_sample_cutoff: int = 10, validation_size: float = 0.1,
                 normalize_weights: bool = True, path_prefix: str = "") -> None:
        self._path_prefix = path_prefix
        self._info_path = info_path
        self._column_path = column_path
        self.train_sample_cutoff = train_sample_cutoff
        self._validation_size = max(0, min(1, validation_size)) # validation size in percentage
        self._normalize_weights = normalize_weights

        with open(self._column_path, "r") as f:
            self._training_columns = f.read().split(",")

        self._data = pd.read_csv(self._info_path)
        self._data["sample_count"] = self._data["sample_count"].astype(int)
        self._data["class_mapping"] = self._data["class_mapping"].apply(eval)

        self._data_train = self._data[self._data["is_train"]].reset_index(drop=True)
        self._data_test = self._data[~self._data["is_train"]].reset_index(drop=True)
        
        # Each task has its own index -> position in matrix later
        self._data_train = self._data_train.reset_index().rename(columns={"index": "task_index"})
        self._data_test = self._data_test.reset_index().rename(columns={"index": "task_index"})

        self._construct_train_matrices()
        self._construct_test_matrices()

    def _construct_train_matrices(self) -> None:
        groups = self._data_train.groupby("name")

        # Define matrix sizes
        self.n_columns = len(self._training_columns) # aka size of input vector
        self.n_samples = int(groups["sample_count"].mean().sum())
        self.n_tasks = len(self._data_train) # aka size of output vector

        self._X = np.zeros((self.n_samples, self.n_columns), dtype="float32") # input data
        self._Y = np.zeros((self.n_samples, self.n_tasks), dtype="float32") # output data
        self._W = np.zeros((self.n_samples, self.n_tasks), dtype="float32") # weight matrix, tells us which outputs to take into account

        row_idx = 0
        self.validation_groups = list()
        for _, group in groups:
            group = group.reset_index(drop=True)
            tmp = pd.read_csv(self._path_prefix+group["file_location"][0])
            group_size = group["sample_count"][0]

            self.validation_groups.append(list(range(row_idx, row_idx+group_size)))

            x = tmp[self._training_columns].to_numpy().astype("float32") # input data
            self._X[row_idx:(row_idx+group_size), :] = x
            for _, row in group.iterrows():
                class_mapping = row["class_mapping"]
                task_y = tmp[row["target_column"]].astype("str").apply(lambda x: class_mapping[x]).to_numpy("float32")
                self._Y[row_idx:(row_idx+group_size), row["task_index"]] = task_y # output data
                self._W[row_idx:(row_idx+group_size), row["task_index"]] = 1 # to which task this output belongs

            row_idx += group_size

        if self._normalize_weights:
            # Some samples are part of more tasks and therefore contribute more to loss -> normalize rows of W
            self._W = self._W / self._W.sum(1)[:, np.newaxis]
        
        # Randomly select some indices from each data set for validation
        self.validation_split()

        self._X = torch.Tensor(self._X)
        self._Y = torch.Tensor(self._Y)
        self._W = torch.Tensor(self._W)

    def _construct_test_matrices(self) -> None:
        groups = self._data_test.groupby("name")

        # Define matrix sizes
        n_columns = len(self._training_columns) # aka size of input vector
        n_samples = int(groups["sample_count"].mean().sum())
        n_tasks = len(self._data_test) # aka size of output vector

        _X = np.zeros((n_samples, n_columns), dtype="float32") # input data
        _Y = np.zeros((n_samples, n_tasks), dtype="float32") # output data
        _W = np.zeros((n_samples, n_tasks), dtype="float32") # weight matrix, tells us which outputs to take into account

        row_idx = 0
        dataset_idx = list()
        dataset_idx_to_name = dict()
        target_val = list()
        for i, (_, group) in enumerate(groups):
            group = group.reset_index(drop=True)
            tmp = pd.read_csv(self._path_prefix+group["file_location"][0])
            group_size = group["sample_count"][0]

            x = tmp[self._training_columns].to_numpy().astype("float32") # input data
            _X[row_idx:(row_idx+group_size), :] = x
            dataset_idx += [i]*group_size
            dataset_idx_to_name[i] = group["name"][0]
            for _, row in group.iterrows():
                class_mapping = row["class_mapping"]
                task_y = tmp[row["target_column"]].astype("str").apply(lambda x: class_mapping[x]).to_numpy("float32")
                _Y[row_idx:(row_idx+group_size), row["task_index"]] = task_y # output data
                _W[row_idx:(row_idx+group_size), row["task_index"]] = 1 # to which task this output belongs
            if len(group) == 1:
                target_val += list(task_y)
            else:
                target_val += [2]*group_size
            row_idx += group_size

        if self._normalize_weights:
            # Some samples are part of more tasks and therefore contribute more to loss -> normalize rows of W
            _W = _W / _W.sum(1)[:, np.newaxis]

        _X = torch.Tensor(_X)
        _Y = torch.Tensor(_Y)
        _W = torch.Tensor(_W)
        self.test = Dataset.TestInfo(
            n_columns, n_samples, n_tasks, _X, _Y, _W,
            dataset_idx, dataset_idx_to_name, target_val
        )

    def to(self, device):
        self._X = self._X.to(device)
        self._Y = self._Y.to(device)
        self._W = self._W.to(device)
        self.test._X = self.test._X.to(device)
        self.test._Y = self.test._Y.to(device)
        self.test._W = self.test._W.to(device)

    def validation_split(self, seed=None):
        if seed:
            random.seed(seed)
        val_idx = [random.sample(l, int(len(l)*self._validation_size)) for l in self.validation_groups]
        val_idx = np.sort(np.array([idx for group in val_idx for idx in group]))
        self._val_idx = np.zeros((self.n_samples,), dtype=bool)
        self._val_idx[val_idx] = True

    def get_train_dataloaders(self, **kwargs):
        # add some default kwargs if not provided
        train_kwargs = {**{"batch_size": 64, "shuffle": True}, **kwargs}
        val_kwargs = {**{"batch_size": 64, "shuffle": False}, **kwargs}
        #_add_to_dict(kwargs, "pin_memory", True) # faster loading from CPU to GPU

        dataset_val = TensorDataset(self._X[self._val_idx], self._Y[self._val_idx], self._W[self._val_idx])
        dataset_train = TensorDataset(self._X[~self._val_idx], self._Y[~self._val_idx], self._W[~self._val_idx])
        return DataLoader(dataset_train, **train_kwargs), DataLoader(dataset_val, **val_kwargs)

    def get_test_dataloaders(self, **kwargs):
        # add some default kwargs if not provided
        test_kwargs = {**{"batch_size": 64, "shuffle": True}, **kwargs}

        dataset_test = TensorDataset(self.test._X, self.test._Y, self.test._W)
        return DataLoader(dataset_test, **test_kwargs)
