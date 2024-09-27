import os
from time import perf_counter

import torch.cuda as cuda

from geo_dataset import Dataset
from geo_models import MultiTask
from geo_train import Trainer
from geo_test import Tester


tic = perf_counter()
def clock() -> str:
    seconds = perf_counter()-tic
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_model_name(model):
    return model.__class__.__name__


def main():
    dataset = Dataset(info_path="data/GEO_v2/training_data_v3.csv")
    print(f"[{clock()}] Dataset loaded")
    trainer = Trainer(device="cuda" if cuda.is_available() else "cpu")
    print(f"[{clock()}] Trainer ready ({trainer._torch_device})")
    tester = Tester(device="cuda" if cuda.is_available() else "cpu")
    print(f"[{clock()}] Tester ready ({tester._torch_device})")

    for latent_dim in [64, 32, 16, 12, 10, 8, 7, 6, 5, 4]:
        for i in range(10): # run training 10 times
            model = MultiTask(encoder_layers=[512, 512], latent_dim=latent_dim,
                                num_tasks=dataset.n_tasks)
            save_dir = f"models/{get_model_name(model)}/{latent_dim}/{i}"
            if os.path.isfile(f"{save_dir}/test.csv"):
                print(f"[{clock()}] Skipping '{save_dir}'")
                continue
            os.makedirs(save_dir, exist_ok=True)
            print(f"[{clock()}] Started training '{save_dir}'")
            df = trainer.train(
                model, dataset, epochs=1000,
                earlystopping_args={"save_path": f"{save_dir}/model.pickle", "skip": 100, "patience": 25, "rope": 1e-3},
                optimizer_args={"lr": 5e-5},
                dataloader_args={"batch_size": 64}
            )
            df.to_csv(f"{save_dir}/train.csv", index=False)
            print(f"[{clock()}] Ended training '{save_dir}'")

            print(f"[{clock()}] Started testing '{save_dir}'")
            df = tester.test(model, dataset)
            df.to_csv(f"{save_dir}/test.csv", index=False)
            print(f"[{clock()}] Ended testing '{save_dir}'")


if __name__ == "__main__":
    main()