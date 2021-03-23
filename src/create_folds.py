import pandas as pd
from sklearn import model_selection


def create_folds(data_path):
    df = pd.read_csv(data_path)
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold
    return df


if __name__ == "__main__":
    df_train = create_folds("input/categorical_feature_encoding/train.csv")
    df_train.to_csv("input/categorical_feature_encoding/train_folds.csv", index=False)
