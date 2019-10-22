from autoPyTorch import AutoNetClassification
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 42
random_state = RandomState(seed)

n = 10
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x[:n],
    y[:n],
    test_size=.2,
    random_state=random_state
)

autoPyTorch = AutoNetClassification("tiny_cs",
                                    log_level='info',
                                    max_runtime=30,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(x_train, y_train, validation_split=.125)
y_pred = autoPyTorch.predict(x_test)

print(accuracy_score(y_test, y_pred))
