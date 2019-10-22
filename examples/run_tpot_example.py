from keras.datasets import fashion_mnist
from numpy.random.mtrand import RandomState
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier

seed = 42
random_state = RandomState(seed)

# x, y = load_digits(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     test_size=.2,
#     random_state=random_state
# )

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

s = x_train.shape[1]
x_train = x_train.reshape((len(x_train), s ** 2))
x_test = x_test.reshape((len(x_test), s ** 2))

clf = TPOTClassifier(
    generations=5,
    population_size=50,
    n_jobs=4,
    random_state=seed,
    verbosity=2,
)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
