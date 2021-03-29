import numpy as np
import pandas as pd

from sklearn import preprocessing

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model


def get_model(df, categorical_columns):
    inputs = []
    outputs = []
    for c in categorical_columns:
        num_unique_vals = int(df[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_vals / 2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_vals + 1, embed_dim, name=c)(inp)
        # apply dropout here
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        outputs.append(out)
    x = layers.Concatenate()(outputs)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=y)
    return model


if __name__ == "__main__":
    train = pd.read_csv("../input/categorical_feature_encoding/train.csv")
    test = pd.read_csv("../input/categorical_feature_encoding/test.csv")
    sample = pd.read_csv("../input/categorical_feature_encoding/sample_submission.csv")

    test.loc[:, "target"] = -1
    data = pd.concat([train, test]).reset_index(drop=True)

    features = [f for f in train.columns if f not in ["id", "target"]]

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        data.loc[:, feat] = lbl_enc.fit_transform(data[feat].astype(str).fillna("-1").values)

    train = data[data.target != -1].reset_index(drop=True)
    test = data[data.target == -1].reset_index(drop=True)

    model = get_model(train, features)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit([train.loc[:, f].values for f in features], train.target.values)
