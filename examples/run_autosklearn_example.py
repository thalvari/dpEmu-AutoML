import autosklearn.classification
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 42
random_state = RandomState(seed)

x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=.2,
    random_state=random_state
)

automl = autosklearn.classification.AutoSklearnClassifier(seed=seed, n_jobs=2)
automl.fit(x_train, y_train)
y_pred = automl.predict(x_test)
print(accuracy_score(y_test, y_pred))
