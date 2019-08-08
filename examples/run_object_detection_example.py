import os
import re
import subprocess
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import RandomState

from dpemu import runner
from dpemu.dataset_utils import load_coco_val_2017
from dpemu.ml_utils import run_ml_module_using_cli
from dpemu.plotting_utils import print_results_by_model, visualize_scores
from dpemu.problemgenerator.array import Array
from dpemu.problemgenerator.filters import JPEG_Compression
from dpemu.problemgenerator.series import Series


class Preprocessor:
    def run(self, _, imgs, params):
        img_filenames = params["img_filenames"]

        for i, img_arr in enumerate(imgs):
            img = Image.fromarray(img_arr)
            path_to_img = "tmp/val2017/" + img_filenames[i]
            img.save(path_to_img, "jpeg", quality=100)

        return None, imgs, {}


class YOLOv3Model:

    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, _1, _2, _3):
        path_to_yolov3_weights = "tmp/yolov3-spp_best.weights"
        if not os.path.isfile(path_to_yolov3_weights):
            subprocess.call(["./scripts/get_yolov3.sh"])

        cline = "libs/darknet/darknet detector map data/coco.data tmp/yolov3-spp.cfg tmp/yolov3-spp_best.weights"
        out = run_ml_module_using_cli(cline)

        match = re.search(r"\(mAP@0.50\) = (\d+\.\d+)", out)
        return {"mAP-50": round(float(match.group(1)), 3)}


class AbstractDetectronModel(ABC):

    def __init__(self):
        self.random_state = RandomState(42)

    def run(self, _1, _2, _3):
        path_to_cfg = self.get_path_to_cfg()
        url_to_weights = self.get_url_to_weights()

        cline = f"""libs/Detectron/tools/test_net.py \
            --cfg {path_to_cfg} \
            TEST.WEIGHTS {url_to_weights} \
            NUM_GPUS 1 \
            TEST.DATASETS '("coco_2017_val",)' \
            MODEL.MASK_ON False \
            OUTPUT_DIR tmp"""
        out = run_ml_module_using_cli(cline)

        match = re.search(r"IoU=0.50      \| area=   all \| maxDets=100 ] = (\d+\.\d+)", out)
        return {"mAP-50": round(float(match.group(1)), 3)}

    @abstractmethod
    def get_path_to_cfg(self):
        pass

    @abstractmethod
    def get_url_to_weights(self):
        pass


class FasterRCNNModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/35858015/12_2017_baselines/"
            "e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml.01_40_54.1xc565DE/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
        )


class MaskRCNNModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/36494496/12_2017_baselines/"
            "e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml.07_50_11.fkwVtEvg/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
        )


class RetinaNetModel(AbstractDetectronModel):
    def __init__(self):
        super().__init__()

    def get_path_to_cfg(self):
        return "libs/Detectron/configs/12_2017_baselines/retinanet_X-101-64x4d-FPN_1x.yaml"

    def get_url_to_weights(self):
        return (
            "https://dl.fbaipublicfiles.com/detectron/36768875/12_2017_baselines/"
            "retinanet_X-101-64x4d-FPN_1x.yaml.08_34_37.FSXgMpzP/output/train/"
            "coco_2014_train%3Acoco_2014_valminusminival/retinanet/model_final.pkl"
        )


def get_err_root_node():
    err_node = Array()
    err_root_node = Series(err_node)
    # err_node.addfilter(GaussianNoise("mean", "std"))
    # err_node.addfilter(Blur_Gaussian("std"))
    # err_node.addfilter(Snow("snowflake_probability", "snowflake_alpha", "snowstorm_alpha"))
    # err_node.addfilter(FastRain("probability", "range_id"))
    # err_node.addfilter(StainArea("probability", "radius_generator", "transparency_percentage"))
    err_node.addfilter(JPEG_Compression("quality"))
    # err_node.addfilter(ResolutionVectorized("k"))
    # err_node.addfilter(Identity())
    return err_root_node


def get_err_params_list():
    # return [{"mean": 0, "std": std} for std in [10 * i for i in range(0, 4)]]
    # return [{"std": std} for std in [i for i in range(0, 4)]]
    # return [{"snowflake_probability": p, "snowflake_alpha": .4, "snowstorm_alpha": 0}
    #                    for p in [10 ** i for i in range(-4, 0)]]
    # return [{"probability": p, "range_id": 255} for p in [10 ** i for i in range(-4, 0)]]
    # return [
    #     {"probability": p, "radius_generator": GaussianRadiusGenerator(0, 50), "transparency_percentage": 0.2}
    #     for p in [10 ** i for i in range(-6, -2)]]
    return [{"quality": q} for q in [10, 20, 30, 100]]
    # return [{"k": k} for k in [1, 2, 3, 4]]
    # return [{}]


def get_model_params_dict_list():
    return [
        {"model": FasterRCNNModel, "params_list": [{}]},
        {"model": MaskRCNNModel, "params_list": [{}]},
        {"model": RetinaNetModel, "params_list": [{}]},
        {"model": YOLOv3Model, "params_list": [{}]},
    ]


def visualize(df):
    # visualize_scores(df, ["mAP-50"], [True], "std", "Object detection with Gaussian noise", x_log=False)
    # visualize_scores(df, ["mAP-50"], [True], "std", "Object detection with Gaussian blur", x_log=False)
    # visualize_scores(df, ["mAP-50"], [True], "snowflake_probability", "Object detection with snow filter", x_log=True)
    # visualize_scores(df, ["mAP-50"], [True], "probability", "Object detection with rain filter", x_log=True)
    # visualize_scores(df, ["mAP-50"], [True], "probability", "Object detection with added stains", x_log=True)
    visualize_scores(df, ["mAP-50"], [True], "quality", "Object detection with JPEG compression", x_log=False)
    # visualize_scores(df, ["mAP-50"], [True], "k", "Object detection with reduced resolution", x_log=False)
    plt.show()


def main():
    imgs, _, _, img_filenames = load_coco_val_2017()

    df = runner.run(
        train_data=None,
        test_data=imgs,
        preproc=Preprocessor,
        preproc_params={"img_filenames": img_filenames},
        err_root_node=get_err_root_node(),
        err_params_list=get_err_params_list(),
        model_params_dict_list=get_model_params_dict_list(),
        n_processes=1
    )

    print_results_by_model(df, ["show_imgs", "mean", "radius_generator", "transparency_percentage", "range_id",
                                "snowflake_alpha", "snowstorm_alpha"])
    visualize(df)


if __name__ == "__main__":
    main()
