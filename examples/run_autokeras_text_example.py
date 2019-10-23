import autokeras as ak
import numpy as np
from autokeras import TextClassifier
from keras.datasets import reuters
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dpemu.utils import generate_tmpdir

seed = 42
random_state = RandomState(seed)

(x_train, y_train), (x_test, y_test) = reuters.load_data(seed=seed)
print(x_train.shape)
print(x_test.shape)
print(len(x_train[0]))
print(len(x_train[1]))
print(len(x_train[2]))
print(len(x_train[2000]))

clf = TextClassifier(verbose=True)
# clf.fit(x=x_train, y=y_train, time_limit=60*30)
#
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_true=y_test, y_pred=y_pred))
# best_pipeline = str(clf.cnn.best_model.produce_model())
# print(best_pipeline)