import argparse
import os
import pickle
from typing import Dict, List, Set

import numpy as np
import pandas as pd
#pd.set_option("display.max_rows", 500)
#pd.set_option("display.max_columns", 500)

from manual_inspect_helper import Input, GEOTask, _warp_string, _print_green, _print_red, _print
from manual_inspect_helper import ACCEPTED_TARGET_COLUMNS, REJECTED_TARGET_COLUMNS


class GEOInspector:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.MAIN_FOLDER = os.path.abspath(self.args.folder)
        self.INFO_PATH = os.path.join(self.MAIN_FOLDER, "info.csv")
        self.ABSTRACTS_PATH = os.path.join(self.MAIN_FOLDER, "abstracts.pickle")
        self.CANDIDATE_PATH = os.path.join(self.MAIN_FOLDER, self.args.inspect_file+".csv")
        self.INSPECTED_PATH = os.path.join(self.MAIN_FOLDER, self.args.inspect_file+".inspected")

        print(f"   MAIN folder: {self.MAIN_FOLDER}")
        print(f"     INFO file: {self.INFO_PATH}")
        print(f"ABSTRACTS file: {self.ABSTRACTS_PATH}")
        print(f"CANDIDATE file: {self.CANDIDATE_PATH}")
        print(f"INSPECTED file: {self.INSPECTED_PATH}")

        self.abstract_dict: Dict[str, str]
        self.inspect_df: pd.DataFrame
        self.info_df: pd.DataFrame
        self.inspected_datasets: Set[str]

        self._get_info()
        self._get_abstracts()
        self._get_candidates()
        self._get_already_inspected()
        #print(self.inspect_df.head())
        #print(self.inspect_df.tail())
        print()


    def _get_abstracts(self):
        with open(self.ABSTRACTS_PATH, "rb") as file:
            self.abstract_dict = pickle.load(file)
        print("Abstracts loaded")


    def _get_info(self):
        self.info_df = pd.read_csv(self.INFO_PATH)
        self.info_df["true_class_distribution"] = self.info_df["true_class_distribution"].apply(eval)
        print("Datasets info loaded")


    def _get_candidates(self):
        if os.path.exists(self.CANDIDATE_PATH):
            self.inspect_df = pd.read_csv(self.CANDIDATE_PATH, encoding="utf-8")
            print(f"Existing '{self.CANDIDATE_PATH}' file detected, continuing")
        else:
            self.inspect_df = pd.DataFrame(columns=self.info_df.columns)
            self.inspect_df["class_mapping"] = None
            self.inspect_df = self.inspect_df.drop(columns=["true_class_distribution"], errors="ignore")
            print(f"No '{self.CANDIDATE_PATH}' file found, starting from scratch")


    def _get_already_inspected(self):
        if os.path.exists(self.INSPECTED_PATH):
            with open(self.INSPECTED_PATH, "r") as f:
                self.inspected_datasets = set(f.read().strip().split(","))
            print(f"Existing '{self.INSPECTED_PATH}' file detected, continuing")
        else:
            self.inspected_datasets = set()
            print(f"No '{self.INSPECTED_PATH}' file found, starting from scratch")


    def _print_dataset_info(self, dataset_name, n, m):
        dataset_info = self.info_df[self.info_df["name"] == dataset_name]
        title = _warp_string(f"{dataset_name} - "+dataset_info["title"].values[0])
        description = _warp_string(dataset_info["description"].values[0])
        abstract_keys = dataset_info["pubmed_id"].values[0]
        if isinstance(abstract_keys, float):
            abstract = ""
        else:
            abstract_keys = [abstract_keys] if len(abstract_keys) == 8 else abstract_keys[1:-1].replace("'", "").split(", ")
            abstracts = [_warp_string(self.abstract_dict[akey]) for akey in abstract_keys]
            abstract = "\n\n".join(abstracts)
        sample_count = dataset_info["sample_count"].values[0]
        value_type = dataset_info["value_type"].values[0]
        num_L1000_genes = dataset_info["num_L1000_genes"].values[0]

        print(f"\n{'='*34} {n:4d}/{m:4d} {'='*35}\n\n{title}\n\n{description}\n\n{abstract}\n")
        print(f"Samples: {sample_count}\nValue type: {value_type}\nL1000 columns: {num_L1000_genes}\n{'='*80}\n")


    def _print_tasks_info(self, tasks: List[GEOTask], force_all: bool = False):
        for i, task in enumerate(tasks):
            if task.accept:
                print(f"[{i}] \033[92m{task.target_column:40s} ACCEPT\033[00m")
            else:
                print(f"[{i}] \033[91m{task.target_column:40s} DENY\033[00m")
            if not force_all and task.target_column in REJECTED_TARGET_COLUMNS:
                continue
            for subtask in task.class_order:
                print(f"    {subtask} -> {task.mapping[subtask]} [{task.class_distribution[subtask]}]")
        print()


    def _print_tasks_correlation(self, dataset_target: pd.DataFrame, tasks: List[GEOTask]):
        tmp = pd.DataFrame()
        column_widths = []
        for i, task in enumerate(tasks):
            _data = dataset_target[task.target_column].apply(lambda x: task.mapping[str(x)]).astype("category")
            _name = (task.target_column[:6] if len(task.target_column) > 6 else task.target_column)+"_"+str(i)
            tmp[_name] = _data
            column_widths.append(len(_name)+2)
        #print(tmp.corr(numeric_only=False).round(decimals=3))

        txt = tmp.corr(numeric_only=False).round(decimals=3).to_string().split("\n")
        column_widths.insert(0, len(txt[0])-len(txt[0].lstrip()) - 2)

        for row, line in enumerate(txt, -1):
            width_so_far = 0
            for col, width in enumerate(column_widths, -1):
                to_print = line[width_so_far:width_so_far+width]
                if row == -1 and col == -1:
                    print(to_print, end="")
                elif row == -1 or col == -1: # handle column and row labels
                    id = max(row, col)
                    if tasks[id].accept:
                        _print_green(to_print)
                    else:
                        _print_red(to_print)
                else: # handle correlation data
                    c = tasks[row].accept + tasks[col].accept
                    (_print, _print, _print_green)[c](to_print)
                width_so_far += width
            print()


    def _validate_user_input(self, command: Input, id: int, remap: List[int]):
        if command in {Input.TASK_SEE, Input.DATASET_ACCEPT, Input.DATASET_REJECT, Input.EXIT, Input.COMMAND_HELP} and id is not None:
            command = Input.COMMAND_UNKNOWN
        elif command in {Input.TASK_ACCEPT, Input.TASK_REJECT, Input.TASK_SPLIT, Input.TASK_MERGE} and remap != []:
            command = Input.COMMAND_UNKNOWN
        return command, id, remap


    def _get_user_input(self):
        cmd = input("|> ")
        cmd = cmd.split(" ")

        try:
            command = Input(cmd[0])
        except ValueError:
            command = Input.COMMAND_UNKNOWN
        id, remap = None, []

        try:
            if len(cmd) > 1:
                id = int(cmd[1])
            if len(cmd) > 2:
                remap = [int(x) for x in cmd[2:]]
        except:
            command = Input.COMMAND_UNKNOWN

        return self._validate_user_input(command, id, remap)


    def _task_accept(self, tasks: List[GEOTask], id: int):
        try:
            tasks[id].accept = True
        except:
            print(f"Task with ID {id} does not exits!")


    def _task_reject(self, tasks: List[GEOTask], id: int):
        try:
            tasks[id].accept = False
        except:
            print(f"Task with ID {id} does not exits!")


    def _task_remap(self, tasks: List[GEOTask], id: int, remap: List[int]):
        if len(tasks) < id+1:
            print(f"Task with ID {id} does not exits!")
            return
        if len(tasks[id].class_order) != len(remap):
            print(f"Number of remap values ({len(remap)}) is not the same as number of target values ({len(tasks[id].class_order)})!")
            return
        tasks[id].mapping = {x: i for i, x in zip(remap, tasks[id].class_order)}


    def _task_split(self, tasks: List[GEOTask], id: int):
        try:
            task_to_split = tasks.pop(id)
            for cls in task_to_split.class_order:
                task = task_to_split.copy()
                task.mapping = {k: (1 if k == cls else 0) for k in task.class_order}
                tasks.insert(id, task)
        except:
            print(f"Task with ID {id} does not exits!")


    def _task_merge(self, tasks: List[GEOTask], id: int):
        try:
            target_column = tasks[id].target_column
            tasks[id].mapping = {x: i for i, x in enumerate(tasks[id].class_order)}
            ids = [i for i, task in enumerate(tasks) if task.target_column == target_column and i != id]
            ids.sort(reverse=True)
            for i in ids:
                tasks.pop(i)
        except:
            print(f"Task with ID {id} does not exits!")


    def _dataset_accept(self, tasks: List[GEOTask]):
        """ accepted = np.array([i for i, task in enumerate(tasks) if task.accept])
        if accepted.size == 0: # nothing to accept
            return
        selected_datasets = dataset_info.loc[dataset_info.index[accepted]]
        selected_datasets["class_mapping"] = [task.mapping for task in tasks if task.accept]
        self.inspect_df = pd.concat([self.inspect_df, selected_datasets], ignore_index=True).reset_index(drop=True)
        self.inspect_df.to_csv(self.CANDIDATE_PATH, index=False, encoding="utf-8") """
        """ for task in tasks:
            if task.accept:
                self.inspect_df = self.inspect_df.append(task.to_series(), ignore_index=True) """
        tmp = pd.DataFrame([task.to_series() for task in tasks if task.accept])
        self.inspect_df = pd.concat([self.inspect_df, tmp], ignore_index=True)
        self.inspect_df.to_csv(self.CANDIDATE_PATH, index=False, encoding="utf-8")
        #print(self.inspect_df.tail())


    def _dataset_done(self, dataset_name: str):
        self.inspected_datasets.add(dataset_name)
        with open(self.INSPECTED_PATH, "w") as f:
            f.write(",".join(self.inspected_datasets))


    def inspection_loop(self):
        """
        determine where we left of if existing inspect file exists
        then for each dataset do:
        - list its name, title, short description, abstract + other info from info_df
        - list all possible tasks
        - for each task accept it, remove it, or change class mapping
        - save results in the inspected file
        - finally move to next dataset or stop inspecting for the time being
        """

        # determine for which datasets we still need to perform manual inspection
        remaining_datasets: set
        if not self.inspect_df.empty:
            already_inspected_datasets = set(self.inspect_df["name"].tolist()) # set takes care of unique values
            remaining_datasets = set(self.info_df["name"].tolist()) - already_inspected_datasets
        else:
            remaining_datasets = set(self.info_df["name"].tolist())
        remaining_datasets -= self.inspected_datasets # remove already inspected datasets

        Input.print_commands()

        # loop over all remaining datasets
        num_to_inspect = len(remaining_datasets)
        for idx, dataset_name in enumerate(remaining_datasets):
            dataset_info = self.info_df[self.info_df["name"] == dataset_name]

            # HARDCODED: reject some undesirable datasets from the start
            if dataset_info["value_type"].values[0] not in {"transformed count"}:
                continue

            # print the relevant information
            self._print_dataset_info(dataset_name, idx, num_to_inspect)

            # get tasks representation
            """ tasks = [
                GEOTask(row["target_column"], row["num_classes"], eval(row["true_class_distribution"]))
                for _, row in dataset_info.iterrows()
            ]
            # calculate default mappings, default acceptance
            for task in tasks:
                task.class_order = [x for x in task.class_distribution]
                task.mapping = {x: i for i, x in enumerate(task.class_order)}
                task.accept = task.target_column in ACCEPTED_TARGET_COLUMNS """
            tasks = [GEOTask.from_series(row) for _, row in dataset_info.iterrows()]
            for task in tasks:
                task.accept = task.target_column in ACCEPTED_TARGET_COLUMNS
            # print tasks info
            self._print_tasks_info(tasks)

            dataset_target = pd.read_csv(dataset_info["file_location"].values[0], usecols=dataset_info["target_column"].tolist())
            self._print_tasks_correlation(dataset_target, tasks)

            while True:
                cmd, id, remap = self._get_user_input()

                if cmd == Input.EXIT:
                    exit()
                elif cmd == Input.TASK_ACCEPT:
                    self._task_accept(tasks, id)
                elif cmd == Input.TASK_REJECT:
                    self._task_reject(tasks, id)
                elif cmd == Input.TASK_REMAP:
                    self._task_remap(tasks, id, remap)
                elif cmd == Input.TASK_SPLIT:
                    self._task_split(tasks, id)
                elif cmd == Input.TASK_MERGE:
                    self._task_merge(tasks, id)
                elif cmd == Input.TASK_SEE:
                    self._print_tasks_info(tasks, True)
                    self._print_tasks_correlation(dataset_target, tasks)
                elif cmd == Input.DATASET_ACCEPT:
                    # we save the current tasks, advance to the next dataset
                    self._dataset_accept(tasks)
                    break
                elif cmd == Input.DATASET_REJECT:
                    # just advance to the next dataset
                    break
                elif cmd == Input.COMMAND_HELP:
                    Input.print_commands()
                else:
                    print("Command cannot be parsed! Type 'hlp' for help")

            self._dataset_done(dataset_name)

        print("All datasets have been inspected.")


def main():
    parser = argparse.ArgumentParser("GEO inspect", "Helps with the inspection of the downloaded GEO datasets.")
    parser.add_argument("-i", "--inspect-file", dest="inspect_file")
    parser.add_argument("-f", "--folder", default="data/GEO_v2", dest="folder")
    args = parser.parse_args()

    args.inspect_file = "candidates"

    inspector = GEOInspector(args)
    inspector.inspection_loop()


if __name__ == "__main__":
    # meant to run from root of the project folder
    main()