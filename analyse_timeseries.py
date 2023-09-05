import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import constants


def main(
    FOLDER,
    POS_INDEX,
    NEG_INDEX,
    SAVE_DIR,
):
    FOLDER = Path(FOLDER)

    # Create the save directory if it does not exist
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    X = []
    y = []

    # Load the data
    for class_target in [0, 1]:
        for file in FOLDER.glob(f"{class_target}/*.npy"):
            X.append(np.load(file))
            y.append(class_target)

    X = np.array(X)
    y = np.array(y)

    # Set numpy seed for reproducibility
    np.random.seed(42)

    # Exploration
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Plot timeseries for each class
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Example of timeseries for each class")
    axs = axs.flatten()

    index = np.where(y == 0)[0][NEG_INDEX]
    sns.lineplot(ax=axs[0], data=X[index])
    axs[0].set_title(constants.LABELS_DICT[0])
    axs[0].get_legend().remove()
    axs[0].set_xticks([])
    axs[0].set_ylim([-0.5, 1])

    index = np.where(y == 1)[0][POS_INDEX]
    sns.lineplot(ax=axs[1], data=X[index])
    axs[1].set_title(constants.LABELS_DICT[1])
    axs[1].get_legend().remove()
    axs[1].set_xticks([])
    axs[1].set_ylim([-0.5, 1])
    plt.tight_layout()
    plt.savefig(str(SAVE_DIR / "timeseries.pdf"))
    # plt.show()

    # We also notice that the classes are in order:
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=y)
    plt.title("y")
    plt.ylabel("Class")
    plt.xlabel("Element")
    plt.tight_layout()
    plt.savefig(str(SAVE_DIR / "order.pdf"))
    # plt.show()

    # Plot the distribution of the number of frames per video
    plt.figure(figsize=(15, 6))
    counts = {
        constants.LABELS_DICT[k]: np.count_nonzero(y == k)
        for k in constants.LABELS_DICT.keys()
    }
    sns.barplot(x=list(counts), y=list(counts.values()))
    plt.title("Elements per class")
    plt.tight_layout()
    plt.savefig(str(SAVE_DIR / "distribution.pdf"))
    # plt.show()
    print(counts)

    # Plot the boxplot of the feature values
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=[X[:, :, f].flatten() for f in range(X.shape[-1])])
    plt.title("Features")
    # plt.show()

    print("Global mean:", np.around(np.mean(X[:, :]), 5))
    print("Global std:", np.around(np.std(X[:, :]), 5))
    print("Global median:", np.around(np.median(X[:, :]), 5))
    print("Global min:", np.around(np.min(X[:, :]), 5))
    print("Global max:", np.around(np.max(X[:, :]), 5))
    for f in range(X.shape[-1]):
        print(
            f"Feature {f}: mean={np.around(np.mean(X[:, :, f]), 5)}, std={np.around(np.std(X[:, :, f]), 5)}, median={np.around(np.median(X[:, :, f]), 5)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse timeseries dataset")

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing the dataset",
    )

    parser.add_argument(
        "--pos_index",
        type=int,
        default=1,
        help="Index of the positive class",
    )

    parser.add_argument(
        "--neg_index",
        type=int,
        default=1,
        help="Index of the negative class",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Folder to save the plots",
    )

    args = parser.parse_args()

    main(args.folder, args.pos_index, args.neg_index, args.save_dir)
