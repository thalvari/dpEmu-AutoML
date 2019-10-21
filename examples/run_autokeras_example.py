import autokeras as ak
import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from dpemu.utils import generate_tmpdir

seed = 42
random_state = RandomState(seed)
n = 50

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digits["data"][:n],
    digits["target"][:n],
    test_size=.2,
    random_state=random_state
)
x_train = x_train.reshape((len(x_train), 8, 8)).astype(np.uint8)
x_test = x_test.reshape((len(x_test), 8, 8)).astype(np.uint8)
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

clf = ak.ImageClassifier(augment=False, path=generate_tmpdir(), verbose=True)
clf.fit(x_train, y_train, time_limit=15)

y_pred = clf.predict(x_test)
print(accuracy_score(y_true=y_test, y_pred=y_pred))
best_pipeline = str(clf.cnn.best_model.produce_model())
print(best_pipeline)
