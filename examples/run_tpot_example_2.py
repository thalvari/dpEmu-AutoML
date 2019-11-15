import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import joblib
import numpy as np
from distributed import Client
from keras.datasets import fashion_mnist
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

if __name__ == '__main__':
    seed = 42
    random_state = RandomState(seed)

    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=.1,
        test_size=.1,
        random_state=random_state
    )
    x_train = x_train.astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # s = x_train.shape[1]
    # x_train = x_train.reshape((len(x_train), s ** 2))
    # x_test = x_test.reshape((len(x_test), s ** 2))

    client = Client()

    clf = TPOTClassifier(
        max_time_mins=1,
        max_eval_time_mins=1,
        n_jobs=-1,
        random_state=seed,
        verbosity=1,
    )
    with joblib.parallel_backend("dask"):
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    print([step[1] for step in clf.fitted_pipeline_.steps])
