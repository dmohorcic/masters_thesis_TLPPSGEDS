import pandas as pd


class L1000:

    def __init__(self, path: str = "data/L1000/L1000.tab", path_prefix: str = ""):
        self._data = pd.read_csv(path_prefix+path, sep="\t")
        self._landmark = self._data[self._data["Type"] == "landmark"].reset_index(drop=True)

        self._landmark_list = list(self._landmark["Symbol"])
        self._landmark_list.sort()
        
        self._landmark_set = set(self._landmark["Symbol"])

    def get_landmark_list(self) -> list:
        return self._landmark_list

    def get_landmark_set(self) -> set:
        return self._landmark_set

    def __len__(self) -> int:
        return len(self._landmark_list)
