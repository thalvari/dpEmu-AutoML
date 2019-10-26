import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

from dpemu.filters.image import RotationPIL
from dpemu.nodes import Array
from dpemu.utils import get_project_root


def main():
    # x, y = load_digits(return_X_y=True)
    # shape = (8, 8)
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train.astype(np.float64)
    shape = (28, 28)

    k = 1
    x = x_train[k]
    y = y_train[k]

    max_val = np.amax(x_train)
    # std_steps = np.round(np.linspace(0, max_val, num=2), 3)
    max_angle_steps = np.round(np.linspace(0, 180, num=2), 3)

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for i, ax in enumerate(axs.reshape(-1)):
        if i % 2 == 0:
            img_node = Array(reshape=shape)
            img_node.addfilter(RotationPIL("max_angle"))
            # img_node.addfilter(GaussianNoise("mean", "std"))
            # img_node.addfilter(Clip("min_val", "max_val"))
            res = img_node.generate_error(x, {"max_angle": max_angle_steps[i // 2]})
            # res = img_node.generate_error(x, {"mean": 0, "std": std_steps[i // 2], "min_val": 0, "max_val": max_val})
            res = np.round(res)
            ax.imshow(res.reshape(shape), cmap="gray_r")
            ax.axis("off")
            ax.set_title(f"Max angle: {max_angle_steps[i // 2]}")
            # ax.set_title(f"Std: {std_steps[i // 2]}")
        else:
            ax.imshow(x.reshape(shape), cmap="gray_r")
            ax.axis("off")
    fig.suptitle(f"Label: {y}")
    plt.savefig(get_project_root().joinpath("out/gaussian_noise.png"))
    plt.show()


if __name__ == "__main__":
    main()
