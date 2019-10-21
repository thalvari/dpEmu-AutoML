import logging
import os

import numpy as np
import pandas as pd
from ludwig.api import LudwigModel
from numpy.random.mtrand import RandomState
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

seed = 42
random_state = RandomState(seed)

n = 10
digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digits["data"][:n],
    digits["target"][:n],
    test_size=.2,
    random_state=random_state
)

df_train = pd.DataFrame(
    np.concatenate([x_train, y_train.reshape((len(y_train), -1)).astype(int)], axis=1),
    columns=[str(i) for i in range(x_train.shape[1])] + ["label"]
)
df_test = pd.DataFrame(
    np.concatenate([x_test, y_test.reshape((len(y_test), -1)).astype(int)], axis=1),
    columns=[str(i) for i in range(x_test.shape[1])] + ["label"]
)

input_features = [{"name": f"{i}", "type": "numerical", "encoder": "stacked_cnn"} for i in range(x_train.shape[1])]
output_features = [{"name": "label", "type": "category"}]
model_definition = {"input_features": input_features, "output_features": output_features}

model = LudwigModel(model_definition, logging_level=logging.INFO)
train_stats = model.train(
    data_df=df_train,
    gpus=["0"],
    random_seed=seed
)

y_pred = model.predict(df_test)["label_predictions"].values.astype(np.float).astype(int)
print(accuracy_score(y_true=y_test, y_pred=y_pred))
