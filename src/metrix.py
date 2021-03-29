import numpy as np
from sklearn import metrics as skmetrics


class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mae": self._mae,
            "mse": self._mse,
            "rmse": self._rmse,
            "msle": self._msle,
            "rmsle": self._rmsle,
            "r2": self._r2
        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not implemented!")
        else:
            return self.metrics[metric](y_true, y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true, y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true, y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true, y_pred)


class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "auc": self._auc,
            "logloss": self._logloss
        }

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric not implemented!")
        if metric == "auc":
            if y_proba is not None:
                return self.metrics["auc"](y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        if metric == "logloss":
            if y_proba is not None:
                return self.metrics["logloss"](y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    # cm = ClassificationMetrics()
    # t = [0, 1, 0, 0, 1, 0]
    # p = [0, 1, 0, 0, 0, 0]
    # print(cm("accuracy", t, p))
    # print(cm("auc", t, p, p))
    # print(cm("logloss", t, p, t))

    rm = RegressionMetrics()
    t = [0, 1, 0, 0, 1, 0]
    p = [0.1, 0.8, 0, 0.2, 0.9, 0]
    print(rm("mae", t, p))
    print(rm("mse", t, p))
    print(rm("rmse", t, p))
    print(rm("msle", t, p))
    print(rm("rmsle", t, p))
    print(rm("r2", t, p))
