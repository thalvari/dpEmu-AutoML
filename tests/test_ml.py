import json

from PIL import Image

from src.ml.utils import run_ml_script
from src.utils import load_digits_as_npy, generate_unique_path


def test_kmeans_with_analysis():
    path_to_data, path_to_labels = load_digits_as_npy()
    path_to_reduced_data = generate_unique_path("tmp", "npy")
    path_to_fitted_model = generate_unique_path("tmp", "joblib")
    path_to_classes_img = generate_unique_path("tmp", "png")
    path_to_clusters_img = generate_unique_path("tmp", "png")
    path_to_scores = generate_unique_path("tmp", "json")

    run_ml_script("python src/ml/kmeans_model.py {} {} {} {}".format(path_to_data, path_to_labels, path_to_reduced_data,
                                                                     path_to_fitted_model))
    run_ml_script("python src/ml/clustering_analyzer.py {} {} {} {} {} {}".format(path_to_reduced_data, path_to_labels,
                                                                                  path_to_fitted_model,
                                                                                  path_to_classes_img,
                                                                                  path_to_clusters_img, path_to_scores))

    with open(path_to_scores, "r") as file:
        scores = json.load(file)
    Image.open(path_to_classes_img).verify()
    Image.open(path_to_clusters_img).verify()

    assert scores == {"v-meas": "0.919", "ARI": "0.898", "AMI": "0.919", "silhouette": "0.79"}
