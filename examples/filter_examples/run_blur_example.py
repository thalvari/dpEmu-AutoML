import numpy as np
from PIL import Image
from dpemu.nodes import Array
from dpemu.filters.image import Blur


def main():
    img = Image.open("data/landscape.png")
    data = np.array(img)
    root_node = Array()
    # root_node.addfilter(filters.Blur_Gaussian('std'))
    # result = root_node.generate_error(data, {'std': 10.0})
    root_node.addfilter(Blur('repeats', 'radius'))
    result = root_node.generate_error(data, {'repeats': 1, 'radius': 20})
    filtered_img = Image.fromarray(result.astype('uint8'), 'RGB')
    filtered_img.show()


if __name__ == "__main__":
    main()