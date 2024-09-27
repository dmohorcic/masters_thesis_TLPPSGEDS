import argparse
import itertools
import os
import pickle
from collections import defaultdict

from Bio import Entrez
import numpy as np
import pandas as pd
from tqdm import tqdm

from orangecontrib.bioinformatics import geo
import serverfiles

ORANGE_GEO_URL = "https://download.biolab.si/datasets/geo/"

# constants that describe the values of interest and their transformations
VALUES_OF_INTEREST = [
    "name", # str (name of the dataset)
    "title", # str
    "description", # str
    "pubmed_id", # str, list -> int
    "platform", # str (almost unique)
    "platform_technology_type", # str (the type of sequencing technology?)
    "feature_count", # str -> int
    "channel_count", # str -> int (only a few have 2, rest is 1)
    "sample_count", # str -> int
    "value_type", # str (either 'count' or 'transformed count', others are few)
    "reference_series", # str (seems unique, rarely 2 or even 3)
    "order", # str (either 'none' or 'ordered', seems uninportant)
]
TRANSFORMATIONS = {
    "name": str,
    "title": str,
    "description": str,
    "pubmed_id": lambda x: [str(y) for y in x] if isinstance(x, list) else str(x),
    "platform": str,
    "platform_technology_type": str,
    "feature_count": int,
    "channel_count": int,
    "sample_count": int,
    "value_type": str,
    "reference_series": str,
    "order": str,
}

from L1000 import L1000


class GEODownloader:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.MAIN_FOLDER = os.path.abspath(args.folder)
        # check if folder exists, if not create it
        if not os.path.isdir(self.MAIN_FOLDER):
            os.makedirs(self.MAIN_FOLDER)
        self.DATASET_FOLDER = os.path.join(self.MAIN_FOLDER, "datasets")
        if not os.path.isdir(self.DATASET_FOLDER):
            os.mkdir(self.DATASET_FOLDER)

        # set info and abstract files paths
        self.INFO_PATH = os.path.join(self.MAIN_FOLDER, "info.csv")
        self.ABSTRACTS_PATH = os.path.join(self.MAIN_FOLDER, "abstracts.pickle")

        # set the Entrez email
        Entrez.email = args.email

        # get the L1000 genome data
        self.l1000 = L1000(args.L1000)

        print(f"    MAIN folder: {self.MAIN_FOLDER}")
        print(f"DATASETS folder: {self.DATASET_FOLDER}")
        print(f"      INFO file: {self.INFO_PATH}")
        print(f" ABSTRACTS file: {self.ABSTRACTS_PATH}")


    def _get_geo_info(self):
        server_files = serverfiles.ServerFiles(server=ORANGE_GEO_URL)
        data_info = server_files.allinfo()

        # we are interested only in datasets that have "Homo sapiens" as the organism
        classification_datasets = {
            key[0]: val
            for key, val in data_info.items()
            if val["sample_organism"] == "Homo sapiens"
        }
        print(f"We have found {len(classification_datasets)} datasets.")
        return classification_datasets


    def _download_data(self, classification_datasets: dict[str, dict[str: str]]):
        info = defaultdict(list)

        with tqdm(total=len(classification_datasets),
                  desc="Downloaded datasets: ", ncols=100,
                  bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}') as prog_bar:
            for key, val in classification_datasets.items():
                target = key[:-4]
                path = os.path.join(self.DATASET_FOLDER, f"{target}.csv")

                # update progress bar
                prog_bar.set_postfix(dataset=target)

                # download dataset
                table = geo.dataset_download(target)

                # get pandas DataFrames
                xdf, ydf, mdf = table.to_pandas_dfs()

                # get number of non-nan L1000 columns
                xdf_l1000_cols = [col for col in xdf.columns if col in self.l1000.get_landmark_set()]
                num_L1000_cols = xdf[xdf_l1000_cols].notna().all().sum()
                if num_L1000_cols == 0:
                    continue

                # insert missing L1000 columns as np.nan
                missing_L1000_cols = self.l1000.get_landmark_set() - set(xdf.columns)
                xdf = xdf.join(pd.DataFrame({mLc: np.nan for mLc in missing_L1000_cols}, index=xdf.index))

                # keep only L1000 columns
                xdf = xdf[self.l1000.get_landmark_list()]

                # get a de-fragmented frame
                xdf = xdf.copy()

                # merge xdf and mdf
                df = xdf.join(mdf)

                # save dataset
                df.to_csv(path, index=False)

                # extract all classification tasks
                subsets = dict()
                for s in val["subsets"]:
                    if s["type"] not in subsets:
                        subsets[s["type"]] = dict()
                    subsets[s["type"]][s["description"]] = len(s["sample_id"])

                for k, v in subsets.items():
                    true_v = mdf[k].value_counts().to_dict()
                    for voi in VALUES_OF_INTEREST:
                        info[voi].append(TRANSFORMATIONS[voi](val[voi]))
                    info["file_location"].append(os.path.relpath(path))
                    info["num_L1000_genes"].append(num_L1000_cols)
                    info["target_column"].append(k)
                    info["num_classes"].append(len(v))
                    info["class_distribution"].append(v)
                    info["true_class_distribution"].append(true_v)

                # update progress bar
                prog_bar.update(1)

        return pd.DataFrame.from_dict(info)


    def download_geo(self):
        classification_datasets = self._get_geo_info()
        self.info_df = self._download_data(classification_datasets)
        self.info_df.to_csv(self.INFO_PATH, index=False)


    def _fetch_abstracts(self, pub_ids: list[str], retmax: int = 1000) -> dict[str, str]:
        abstract_dict = dict()
        for i in range(0, len(pub_ids), retmax):
            j = i + retmax
            if j >= len(pub_ids):
                j = len(pub_ids)

            handle = Entrez.efetch(db="pubmed", id=",".join(pub_ids[i:j]), rettype="xml", retmode="text", retmax=retmax)
            records = Entrez.read(handle)
            abstracts = ["\n".join(str(x) for x in pubmed_article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])
                        if "Abstract" in pubmed_article["MedlineCitation"]["Article"]
                        else pubmed_article["MedlineCitation"]["Article"]["ArticleTitle"]
                        for pubmed_article in records["PubmedArticle"]]
            abstract_dict.update(zip(pub_ids[i:j], abstracts))

        return abstract_dict


    def download_abstracts(self):
        print("Fetching abstracts from PubMed...", end="", flush=True)
        # extract all unique PubMed IDs, and transform them into a list
        pubmed_ids = self.info_df["pubmed_id"].unique().tolist() # some elements are nested lists
        pubmed_ids = [[x] if len(x) == 8 else x[1:-1].replace("'", "").split(", ") for x in pubmed_ids if isinstance(x, str)]
        pubmed_ids = list(set(itertools.chain.from_iterable(pubmed_ids)))

        abstract_dict = self._fetch_abstracts(pubmed_ids)
        print("OK")

        with open(self.ABSTRACTS_PATH, "wb") as file:
            pickle.dump(abstract_dict, file)


def main():
    parser = argparse.ArgumentParser("GEO download", "Downloads GEO datasets.")
    parser.add_argument("-f", "--folder", default="data/GEO_v2", dest="folder")
    parser.add_argument("-e", "--email", default="dm4492@student.uni-lj.si", dest="email")
    parser.add_argument("-l", "--L1000", default="data/L1000/L1000.tab", dest="L1000")
    args = parser.parse_args()

    geo = GEODownloader(args)
    geo.download_geo()
    geo.download_abstracts()


if __name__ == "__main__":
    main()