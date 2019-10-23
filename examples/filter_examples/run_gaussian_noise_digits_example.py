import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from dpemu.filters.common import GaussianNoise, Clip
from dpemu.nodes import Array
from dpemu.utils import get_project_root


def main():
    x, y = load_digits(return_X_y=True)
    x = x[1]
    y = y[1]

    min_val = np.amin(x)
    max_val = np.amax(x)
    std_steps = np.round(np.linspace(0, max_val, num=8), 3)

    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    for i, ax in enumerate(axs.reshape(-1)):
        img_node = Array(reshape=(8, 8))
        img_node.addfilter(GaussianNoise("mean", "std"))
        img_node.addfilter(Clip("min_val", "max_val"))
        res = img_node.generate_error(x, {"mean": 0, "std": std_steps[i], "min_val": min_val, "max_val": max_val})
        res = np.round(res)
        print(res.dtype)
        print(res)
        ax.imshow(res.reshape((8, 8)), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"Std: {std_steps[i]}")
    fig.suptitle(f"Label: {y}")
    plt.savefig(get_project_root().joinpath("out/gaussian_noise.png"))
    plt.show()


if __name__ == "__main__":
    main()
