import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from geo_models import AutoEncoder, MultiTask
from geo_dataset import Dataset

TYPE = "dataset"  # dataset, target

SIZES = [4, 5, 6, 7, 8, 10, 12, 16, 32, 64]


def main() -> None:
    # read datasets info
    dataset = Dataset(info_path="data/GEO_v2/training_data_v3.csv", normalize_weights=False)

    X = dataset.test._X
    if TYPE == "dataset":
        colors = dataset.test.dataset_idx
    elif TYPE == "target":
        colors = dataset.test.target_val

    # plot raw pca data
    pca = PCA()
    x_pca = pca.fit_transform(X)
    # plt.clf()
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], s=1, c=colors)
    # plt.savefig(f"figures/results/encodings/{TYPE}_pca_l1000.png")
    # with np.printoptions(precision=4, suppress=True):
    #     print("baseline")
    #     print(pca.explained_variance_ratio_)

    ALL_PCA_VARIANCE = dict()
    ALL_PCA_VARIANCE["raw"] = pca.explained_variance_ratio_

    # plot all model encodings
    # read encodedy data
    for model_name in ["AutoEncoder", "MultiTask"]:
        ALL_PCA_VARIANCE[model_name] = dict()
        for size in SIZES:
            explained_variance = np.zeros((10, size))
            for fold in range(10):
                # read model
                if model_name == "AutoEncoder":
                    model = AutoEncoder(encoder_layers=[512, 512], latent_dim=size)
                elif model_name == "MultiTask":
                    model = MultiTask(encoder_layers=[512, 512], latent_dim=size, num_tasks=dataset.n_tasks)
                with open(f"models/{model_name}/{size}/{fold}/model.pickle", "rb") as f:
                    checkpoint = pickle.load(f)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                encoded = model.encoder(X).detach().numpy()
                encoded_pca = pca.fit_transform(encoded)
                # plt.clf()
                # plt.scatter(encoded_pca[:, 0], encoded_pca[:, 1], s=1, c=colors)
                # plt.savefig(f"figures/results/encodings/{TYPE}/{model_name}-{size}-{fold}.png")
                # with np.printoptions(precision=4, suppress=True):
                #     print(model_name, size, fold)
                #     print(pca.explained_variance_ratio_)
                explained_variance[fold, :] = pca.explained_variance_ratio_
            ALL_PCA_VARIANCE[model_name][size] = explained_variance
    with open("figures/results/encodings/explained_variance_ratio.pickle", "wb") as f:
        pickle.dump(ALL_PCA_VARIANCE, f)

if __name__ == "__main__":
    main()
