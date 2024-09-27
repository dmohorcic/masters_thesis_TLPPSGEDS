import os
from collections import defaultdict
import pickle
from time import perf_counter

import pandas as pd
import numpy as np
import torch.cuda as cuda
from tqdm import tqdm

from geo_dataset import Dataset
from geo_test import Tester

tic = perf_counter()
def clock() -> str:
    seconds = perf_counter()-tic
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    dataset = Dataset(info_path="data/GEO_v2/training_data_v3.csv")
    testing_df = dataset._data
    groups = testing_df.groupby("name")
    training_columns = dataset._training_columns
    print(f"[{clock()}] Dataset loaded")
    tester = Tester(device="cpu")
    print(f"[{clock()}] Tester ready ({tester._torch_device})")

    log_metrics = defaultdict(list)

    for name, group in tqdm(groups):
        group = group.reset_index(drop=True)
        tmp = pd.read_csv(dataset._path_prefix+group["file_location"][0])

        x = tmp[training_columns].to_numpy().astype("float64")

        print(f"[{clock()}] Started testing {group['name'][0]}")
        for _, row in group.iterrows():
            class_mapping = row["class_mapping"]
            y = tmp[row["target_column"]].astype("str").apply(lambda x: class_mapping[x]).to_numpy("float64")

            # L1000
            y_pred, y_pred_proba = tester._loocv(x, y)
            log_metrics["name"].append(name)
            log_metrics["target"].append(row["target_column"])
            log_metrics["input"].append("L1000")
            tester._loocv_score(y, y_pred, y_pred_proba, log_metrics)

            # Majority
            y_pred_proba = y.sum()/y.shape[0]
            y_pred = (y_pred_proba > 0.5)*1
            y_pred_proba = np.ones(y.shape) * y_pred_proba
            y_pred = np.ones(y.shape) * y_pred
            log_metrics["name"].append(name)
            log_metrics["target"].append(row["target_column"])
            log_metrics["input"].append("majority")
            tester._loocv_score(y, y_pred, y_pred_proba, log_metrics)

    df = pd.DataFrame.from_dict(log_metrics)
    df.to_csv("data/GEO_v2/test_default_v3.csv", index=False)


if __name__ == "__main__":
    main()
