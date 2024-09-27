from collections import defaultdict
from copy import deepcopy
import pickle
from time import perf_counter

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from geo_dataset import Dataset
from geo_models import BaseGEOModel


__all__ = ["Trainer"]


def _transfer_metrics(source: dict, target: defaultdict, prefix: str):
    for k, v in source.items():
        target[prefix+k].append(v)


class EarlyStopping:

    def __init__(self, patience: int = 15, skip: int = 0, minmax: str = "min",
                 rope: float = 1e-5, model: BaseGEOModel = None,
                 save_path: str = None):

        self.skip = skip
        self.patience = patience
        self.rope = abs(rope)
        self.counter = 0
        self.skip_counter = 0 # also counts epochs

        self.minmax = minmax
        self.comparison_f = (lambda x, y: x < y-self.rope) if self.minmax == "min" else (lambda x, y: x > y+self.rope)
        self.value = float("inf") if self.minmax == "min" else -float("inf")

        self.model = model
        self.save_path = save_path
        self.is_saving = bool(self.save_path and self.model)

        self.successful_comparison = (
            self._save_model if self.is_saving else self._save_model_here
        )

        self.load_model = (
            self._load_model if self.is_saving else self._load_model_here
        )

    def _save_model(self):
        with open(self.save_path, "wb") as f:
            tmp_model = deepcopy(self.model)
            pickle.dump({"epoch": self.skip_counter, "model_state_dict": tmp_model.cpu().state_dict()}, f)
    def _save_model_here(self):
        tmp_model = deepcopy(self.model)
        self._best_model = tmp_model.cpu().state_dict()

    def _load_model(self):
        with open(self.save_path, "rb") as f:
            checkpoint = pickle.load(f)
        # model_state_dict is for a CPU model!
        self.model.cpu()
        self.model.load_state_dict(checkpoint["model_state_dict"])
    def _load_model_here(self):
        self.model.cpu()
        self.model.load_state_dict(self._best_model)

    def reset(self):
        self.counter = 0
        self.skip_counter = 0
        self.value = float("inf") if self.minmax == "min" else -float("inf")

    def __call__(self, value):
        self.skip_counter += 1
        if self.skip_counter < self.skip:
            if self.comparison_f(value, self.value):
                self.value = value
                self.successful_comparison()
            self.counter = 0
            return False

        if self.comparison_f(value, self.value):
            self.value = value
            self.counter = 0
            self.successful_comparison()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


class DummyScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass


class Trainer:

    def __init__(self, device: str = "cpu", verbose: bool = True):
        self._torch_device = device
        self.verbose = verbose


    def train(self, model: BaseGEOModel, dataset: Dataset, epochs: int = 100, *,
              earlystopping_args: dict = {}, optimizer_args: dict = {},
              scheduler_args: dict = {}, dataloader_args: dict = {}
    ) -> pd.DataFrame:
        """
        Training loop.

        Args:
            model (BaseGEOModel): Model to train.
            dataset (Dataset): The dataset.
            epochs (int, optional): Number of epochs. Defaults to 100.
            earlystopping_args (dict, optional): Look at geo_train.EarlyStopping for keyword
            arguments. Defaults to {"save_path": "models/GEO_v2/model.pickle"}.
            optimizer_args (dict, optional): Look at torch.optim.Adam for keyword arguments.
            Defaults to {"lr": 1e-5}.
            scheduler_args (dict, optional): Look at torch.optim.lr_scheduler.MultiStepLR for keyword
            arguments. Defaults to {"use": False, "milestones": [150, 300, 450]}.
            dataloader_args (dict, optional): Look at torch.utils.data.DataLoader for keyword
            arguments. Defaults to {"batch_size": 64, "shuffle": True}.

        Returns:
            pd.DataFrame: train and validation results for each epoch.
        """

        # Update kwargs with defaults
        earlystopping_args = {**{"save_path": "models/GEO_v2/model.pickle"}, **earlystopping_args}
        optimizer_args = {**{"lr": 1e-5}, **optimizer_args}
        scheduler_args = {**{"use": False, "milestones": [150, 300, 450]}, **scheduler_args}
        dataloader_args = {**{"batch_size": 64, "shuffle": True}, **dataloader_args}

        # Setup dataset and dataloaders
        dataset.to(self._torch_device)
        train_dataloader, val_dataloader = dataset.get_train_dataloaders(**dataloader_args)

        # Setup model
        model.to(self._torch_device)
        self._alt_name = model.alt_score_name()

        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), **optimizer_args)
        use_scheduler = scheduler_args.pop("use", False)
        if use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        else:
            scheduler = DummyScheduler()

        # Setup early stopping
        es = EarlyStopping(model=model, **earlystopping_args)
        max_num_of_stops: int = 3
        num_of_stops: int = 0
        es_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(1, max_num_of_stops)), gamma=0.1)

        log_metrics = defaultdict(list)

        # Main training loop
        for epoch in range(1, epochs+1):

            log_metrics["epoch"].append(epoch)

            # TRAIN
            train_metrics = self._train_one_pass(model, train_dataloader, optimizer)

            _transfer_metrics(train_metrics, log_metrics, "train_")

            # VALIDATION
            val_metrics = self._val_one_pass(model, val_dataloader)

            scheduler.step()

            _transfer_metrics(val_metrics, log_metrics, "val_")

            if self.verbose:
                print(f"{epoch}/{epochs} [{train_metrics['time']:.2f}s] LOSS: {train_metrics['loss']:.3f} {self._alt_name.upper()}: {train_metrics['alt_loss']:.3f} | [{val_metrics['time']:.2f}s] LOSS: {val_metrics['loss']:.3f} {self._alt_name.upper()}: {val_metrics['alt_loss']:.3f}")

            # EarlyStopping
            should_stop = es(val_metrics["loss"])
            if should_stop:
                num_of_stops += 1
                if num_of_stops == max_num_of_stops:
                    break

                # just lower learning rate and reset early stopping
                es_scheduler.step()
                es.load_model()
                model.to(self._torch_device)
                tmp_earlystopping_args = {**earlystopping_args}
                tmp_earlystopping_args["skip"] = 0
                es = EarlyStopping(model=model, **tmp_earlystopping_args)
                print("Reducing learning rate")

        df = pd.DataFrame.from_dict(log_metrics)
        df["time"] = df["train_time"] + df["val_time"]

        # restore best model
        es.load_model() # has a pointer, should be ok
        model.to(self._torch_device) # ES restores a cpu version, we cast it back

        return df

    def _train_one_pass(self, model, dataloader, optimizer):
        model.train()
        metrics = {"loss": 0.0, "alt_loss": 0.0}
        tic = perf_counter()
        for x, y, w in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = model.criterion(pred, x, y, w)
            loss.backward()
            optimizer.step()
            metrics["loss"] += loss.item()
            metrics["alt_loss"] += model.alt_score(pred, x, y, w).item()
        toc = perf_counter()
        metrics["time"] = toc-tic
        metrics["loss"] /= len(dataloader)
        metrics["alt_loss"] /= len(dataloader)
        return metrics

    def _val_one_pass(self, model, dataloader):
        model.eval()
        metrics = {"loss": 0.0, "alt_loss": 0.0}
        tic = perf_counter()
        for x, y, w in dataloader:
            pred = model(x)
            loss = model.criterion(pred, x, y, w)
            metrics["loss"] += loss.item()
            metrics["alt_loss"] += model.alt_score(pred, x, y, w).item()
        toc = perf_counter()
        metrics["time"] = toc-tic
        metrics["loss"] /= len(dataloader)
        metrics["alt_loss"] /= len(dataloader)
        return metrics
