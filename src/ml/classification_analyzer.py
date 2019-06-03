import json
import pickle
import sys

import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class ClassificationAnalyzer:

    def __init__(self, paths):
        with open(paths[0], "rb") as file:
            self.data = pickle.load(file)
        self.labels = np.load(paths[1])
        self.fitted_clf = load(paths[2])
        self.path_to_best_clf_params = paths[3]
        self.path_to_scores = paths[4]
        self.path_to_confusion_matrix = paths[5]
        np.random.seed(42)

    def analyze(self):
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=.2,
                                                                            random_state=42)

        predicted_test_labels = self.fitted_clf.predict(test_data)

        scores = {}
        scores["train_data_mean_accuracy"] = self.fitted_clf.score(train_data, train_labels)
        scores["test_data_mean_accuracy"] = np.mean(predicted_test_labels == test_labels)
        scores = {k: round(v, 3) for k, v in scores.items()}
        cm = confusion_matrix(test_labels, predicted_test_labels)

        with open(self.path_to_best_clf_params, "w") as fp:
            json.dump(self.fitted_clf.best_params_, fp)
        with open(self.path_to_scores, "w") as fp:
            json.dump(scores, fp)
        np.save(self.path_to_confusion_matrix, cm)


def main(argv):
    analyzer = ClassificationAnalyzer(argv[1:])
    analyzer.analyze()


if __name__ == "__main__":
    main(sys.argv)