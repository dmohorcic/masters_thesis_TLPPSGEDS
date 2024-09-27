import argparse
import os
import random
from typing import Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import mode as st_mode

from L1000 import L1000


def calculate_intersection(limit: int, geo_L1000_columns: Dict[str, Set[str]]):
    common_cols = None
    for v in geo_L1000_columns.values():
        if len(v) >= limit:
            if common_cols is None:
                common_cols = v
            else:
                common_cols = common_cols.intersection(v)
    return common_cols if common_cols is not None else set()


def main():
    parser = argparse.ArgumentParser("GEO prepare train", "Helps with the preparation of what data to choose for training.")
    parser.add_argument("-i", "--in_file", dest="in_file", default="data/GEO_v2/candidates_manual_fix.csv")
    parser.add_argument("-o", "--out_file", dest="out_file", default="data/GEO_v2/training")
    args = parser.parse_args()

    l1000 = L1000()

    # Read info data
    data = pd.read_csv(args.in_file)
    data["sample_count"] = data["sample_count"].astype(int)
    data["class_distribution"] = data["class_distribution"].apply(eval)
    data["class_mapping"] = data["class_mapping"].apply(eval)

    # Read all non-empty L1000 columns
    geo_L1000_columns = dict()
    for name in tqdm(data["name"].unique(), "Reading L1000 columns"):
        row = data[data["name"] == name].reset_index(drop=True).loc[0]
        tmp = pd.read_csv(row["file_location"], usecols=l1000.get_landmark_list())
        #print(len(tmp.columns))
        nonempty_cols = tmp.notna().all()
        if row["num_L1000_genes"] != nonempty_cols.sum():
            print(f"Problem with dataset {name}: {row['num_L1000_genes']} != {nonempty_cols.sum()}")
        geo_L1000_columns[name] = set(nonempty_cols[nonempty_cols].index) 

    # Ensure that the target class for each task is in minority
    for i, row in data.iterrows():
        dist = row["class_distribution"]
        mapping = row["class_mapping"]
        if sum(v for k, v in dist.items() if mapping[k] == 1) > sum(v for k, v in dist.items() if mapping[k] == 0):
            new_mapping = {k: 1-v for k, v in mapping.items()}
            data.at[i, "class_mapping"] = new_mapping

    # Print some basic stats out of the box
    groups = data.groupby("name")
    samples = groups["sample_count"].mean().to_numpy()
    n_tasks = groups.size().to_numpy()
    print("Number of samples:", int(samples.sum()))
    print("Number of data sets:", len(groups))
    print("Average tasks per dataset:", round(n_tasks.sum() / len(groups), 3))

    num_L1000_genes = groups["num_L1000_genes"].mean().to_numpy()
    print("Number of L1000 genes columns in datasets:")
    print(f"         Mode: {int(st_mode(num_L1000_genes, keepdims=False).mode)}")
    print(f"Mean with std: {np.mean(num_L1000_genes):.1f} +- {np.std(num_L1000_genes):.1f}")

    # Plot number of samples and number of L1000 columns
    x, y, samples = list(), list(), list()
    for limit in range(900, 1000, 1):
        common_cols = calculate_intersection(limit, geo_L1000_columns)
        x.append(limit)
        y.append(len(common_cols))
        samples.append(data[data["num_L1000_genes"] >= limit].groupby("name")["sample_count"].mean().sum())

    fig, ax = plt.subplots()
    ax.plot(x, y, color="red", marker="o")
    ax.set_ylabel("Number of common non-empty\nL1000 genes columns", color="red")
    ax.set_xlabel("Number of L1000 genes columns cutoff")
    ax2 = ax.twinx()
    ax2.plot(x, samples, color="blue", marker="o")
    ax2.set_ylabel("Number of samples", color="blue")
    plt.show()

    # Let the user choose the cutoff coefficient
    cutoff = None
    while not cutoff:
        try:
            cutoff = int(input("Enter L1000 cutoff (default 958): ") or 958)
        except:
            pass

    # We will keep only tasks with more than cutoff L1000 columns
    columns_to_keep = list(calculate_intersection(cutoff, geo_L1000_columns))
    columns_to_keep.sort()
    print(f"We will keep datasets with {len(columns_to_keep)} most common L1000 columns")
    to_train = data[data["num_L1000_genes"] >= cutoff]

    # Keep only datasets with 50 samples or more
    cutoff = None
    while not cutoff:
        try:
            cutoff = int(input("Enter sample cutoff (default 50): ") or 50)
        except:
            pass
    to_train = to_train[to_train["sample_count"] >= cutoff]

    # Split 50-50 into train and test
    random.seed(1)  # this gives roughly equal train-test split
    datasets_names = to_train["name"].unique().tolist()
    train_names = random.sample(datasets_names, k=len(datasets_names)//2)
    test_names = list(set(datasets_names)-set(train_names))
    to_train["is_train"] = to_train["name"].isin(train_names)
    _train_groups = to_train[to_train["name"].isin(train_names)].groupby("name")
    _test_groups = to_train[to_train["name"].isin(test_names)].groupby("name")
    print("Train len:", len(train_names))
    print("Test len:", len(test_names))
    print("Train samples:", _train_groups["sample_count"].mean().sum())
    print("Test samples:", _test_groups["sample_count"].mean().sum())

    # Print some statistics of new dataset
    groups = to_train.groupby("name")
    samples = groups["sample_count"].mean().to_numpy()
    n_tasks = groups.size().to_numpy()
    print("Number of kept samples:", int(samples.sum()))
    print("Number of kept data sets:", len(groups))
    print("Average tasks per kept dataset:", round(n_tasks.sum() / len(groups), 3))

    # Save the new dataset
    to_train.to_csv(f"{args.out_file}_data.csv", index=False)
    with open(f"{args.out_file}_columns.txt", "w") as f:
        f.write(",".join(columns_to_keep))


if __name__ == "__main__":
    main()