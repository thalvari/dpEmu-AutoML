import sys

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from dpemu.filters.image import RotationPIL
from dpemu.nodes import Array, Series
from dpemu.utils import get_project_root


def main():
    max_angle = float(sys.argv[1])
    n = 4

    digits = load_digits()
    x = digits["data"][:n]
    y = digits["target"][:n]

    img_node = Array(reshape=(8, 8))
    root_node = Series(img_node)
    img_node.addfilter(RotationPIL("max_angle"))
    res = root_node.generate_error(x, {"max_angle": max_angle})

    fig, axs = plt.subplots(2, n, constrained_layout=True)
    for i, ax in enumerate(axs[0]):
        ax.imshow(x[i].reshape((8, 8)), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"Label: {y[i]}")
    for i, ax in enumerate(axs[1]):
        ax.imshow(res[i].reshape((8, 8)), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"Label: {y[i]}")
    plt.savefig(get_project_root().joinpath("out/rotation.png"))
    plt.show()


if __name__ == "__main__":
    main()
