from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        :param df: pandas dataframe
        :param categorical_features: list of column names, e.g. ["ord_1", "nom_0", ...]
        :param encoding_type: label, binary, ohe
        :param handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) # array
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = f"{c}__bin__{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        self.ohe = preprocessing.OneHotEncoder()
        self.ohe.fit(self.df[self.cat_feats].values)
        return self.ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood!")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = f"{c}__bin__{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe
        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        else:
            raise Exception("Encoding type can not understand!")


if __name__ == "__main__":
    import pandas as pd
    from sklearn import linear_model
    df_train = pd.read_csv("../input/categorical_feature_encoding/train.csv")
    df_test = pd.read_csv("../input/categorical_feature_encoding/test.csv")
    sample = pd.read_csv("../input/categorical_feature_encoding/sample_submission.csv")
    train_ids = df_train["id"].values
    test_ids = df_test["id"].values

    train_len = len(df_train)
    test_len = len(df_test)
    # print(train_len, test_len)

    df_test["target"] = -1
    df_full = pd.concat([df_train, df_test])

    cols = [c for c in df_train.columns if c not in ["id", "target"]]

    # binary encoding
    # cat_feats = CategoricalFeatures(df_full, categorical_features=cols, encoding_type="binary", handle_na=True)
    # full_data_transformed = cat_feats.fit_transform()
    #
    # train_df = full_data_transformed[full_data_transformed["id"].isin(train_ids)].reset_index(drop=True)
    # test_df = full_data_transformed[full_data_transformed["id"].isin(test_ids)].reset_index(drop=True)

    # one hot encoding
    cat_feats = CategoricalFeatures(df_full, categorical_features=cols, encoding_type="ohe", handle_na=True)
    full_data_transformed = cat_feats.fit_transform()

    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df_train.target.values)
    preds = clf.predict_proba(X_test)[:, 1]

    sample.loc[:, "target"] = preds
    sample.to_csv("submission.csv", index=False)
