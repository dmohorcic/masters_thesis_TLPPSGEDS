from copy import deepcopy
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Union

import pandas as pd

__all__ = [
    "ACCEPTED_TARGET_COLUMNS", "REJECTED_TARGET_COLUMNS", "POSSIBLE_TARGET_COLUMNS",
    "Input", "GEOTask",
    "_warp_string", "_print_red", "_print_green", "_print_yellow", "_print"
]


ACCEPTED_TARGET_COLUMNS = [
    "disease state",
    "infection",
    "genotype/variation",
    "specimen",
    "tissue",
    "cell type",
    "cell line",
    "protocol",
    "agent",
    "gender",
]

REJECTED_TARGET_COLUMNS = [
    "time",
    "age",
    "individual",
    "dose",
]

POSSIBLE_TARGET_COLUMNS = [
    "other",
    "development stage",
    "stress",
    "metabolism",
    "strain",
    "temperature",
    "isolate",
    "growth protocol",
]


class Input(Enum):
    TASK_ACCEPT = "tac" # accept specified task as is
    TASK_REJECT = "tre" # reject specified task
    TASK_SEE = "see" # see all current tasks and their mappings
    TASK_REMAP = "map" # remap the class distribution differently
    TASK_SPLIT = "spl" # split task into n 1-vs-other tasks
    TASK_MERGE = "mrg" # split task into n 1-vs-other tasks
    DATASET_ACCEPT = "dac" # accept the currently changed tasks of the dataset
    DATASET_REJECT = "dre" # reject the dataset
    EXIT = "ext" # stop the inspection, all previous changes are saved
    COMMAND_UNKNOWN = "" # for the unknown commands
    COMMAND_HELP = "hlp"

    @classmethod
    def print_commands(cls):
        print("[tac] <id>                  | Accepth the task with ID <id>")
        print("[tre] <id>                  | Reject the task with ID <id>")
        print("[see]                       | See all the tasks again")
        print("[map] <id> <new task map>   | Remap the task with ID <id> to new map")
        print("[spl] <id>                  | Split the task with ID <id> into n 1-vs-other tasks")
        print("[mrg] <id>                  | Merge the tasks with same target column as ID <id>")
        print("[dac]                       | Accept the current dataset")
        print("[dre]                       | Reject the current dataset")
        print("[ext]                       | Exit the inspection loop")
        print("[hlp]                       | See this list of commands")


@dataclass
class GEOTask:
    name: str = ""
    title: str = ""
    description: str = ""
    pubmed_id: Union[List[int], int, float] = 0
    platform: str = ""
    platform_technology_type: str = ""
    feature_count: int = 0
    channel_count: int = 0
    sample_count: int = 0
    value_type: str = ""
    reference_series: str = ""
    order: str = ""
    file_location: str = ""
    num_L1000_genes: int = 0
    target_column: str = ""
    num_classes: int = 0
    class_distribution: Dict[str, int] = dict # true_class_distribution

    class_order: List[str] = list
    mapping: Dict[str, int] = dict
    accept: bool = False

    def copy(self) -> "GEOTask":
        return deepcopy(self)

    @classmethod
    def from_series(cls, row: pd.Series) -> "GEOTask":
        task = GEOTask(row["name"], row["title"], row["description"], row["pubmed_id"],
                       row["platform"], row["platform_technology_type"], row["feature_count"],
                       row["channel_count"], row["sample_count"], row["value_type"],
                       row["reference_series"], row["order"], row["file_location"],
                       row["num_L1000_genes"], row["target_column"], row["num_classes"],
                       row["true_class_distribution"])
        task.class_order = [x for x in task.class_distribution]
        task.mapping = {x: i for i, x in enumerate(task.class_order)}
        return task

    def to_series(self) -> pd.Series:
        self_dict = asdict(self)
        del self_dict["accept"]
        del self_dict["class_order"]
        self_dict["class_mapping"] = self_dict.pop("mapping")
        return pd.Series(self_dict)


def _warp_string(input_string: str) -> str:
    words = input_string.split()
    lines = []
    current_line = words[0]
    for word in words[1:]:
        if len(current_line) + len(word) + 1 <= 80:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "\n".join(lines)


def _print_red(s: str):
    print("\033[91m"+s+"\033[00m", end="")
def _print_green(s: str):
    print("\033[92m"+s+"\033[00m", end="")
def _print_yellow(s: str):
    print("\033[93m"+s+"\033[00m", end="")
def _print(s: str):
    print(s, end="")