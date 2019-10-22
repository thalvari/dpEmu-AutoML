import h2o
import numpy as np
from h2o.automl import H2OAutoML
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

h2o.init()

random_state = RandomState(42)
digits = load_digits()
train_data, test_data, train_labels, test_labels = train_test_split(
    digits["data"],
    digits["target"],
    test_size=.2,
    random_state=random_state
)

train = h2o.H2OFrame(np.concatenate((train_data, train_labels.reshape(-1, 1)), axis=1))
test = h2o.H2OFrame(np.concatenate((test_data, test_labels.reshape(-1, 1)), axis=1))

x = np.array(train.columns)[:-1].tolist()
y = np.array(train.columns)[-1].tolist()

train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

aml = H2OAutoML(max_models=3, seed=42)
aml.train(x=x, y=y, training_frame=train)

leader_params = aml.leader.get_params()
best_pipeline = [leader_params["model_id"]["actual_value"]["name"]]
if "base_models" in leader_params:
    for base_model in leader_params["base_models"]["actual_value"]:
        best_pipeline.append(base_model["name"])
print(best_pipeline)

test_pred = aml.predict(test).as_data_frame(header=False)["predict"].values.astype(int)
print(np.round(np.mean(test_pred == test_labels), 3))
