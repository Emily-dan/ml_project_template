# ml_project_template

Code by Abhishek Thakur's video

## Cross-Validation types

- KFold
- StratifiedKFold
- Multilabel Classification
- Regression Cross-Validation
- hold-out based validation

## Categorical Variables

- Nominal
  - Nominal variables describe categories that do not have a specific order to them. 
  - Example: [male, female]
- Ordinal
  - Ordinal variables have two are more categories that can be ordered or ranked. 
  - Example: [low, medium, high]

## Binary Classification Metrics

- Accuracy: $\frac{TP+TN}{TP+TN+FP+FN}$
- Precision: $\frac{TP}{TP+FP}$
- Recall: $\frac{TP}{TP+FN}$
- F1-score(F1): $\frac{2RP}{R+P}=\frac{2TP}{2TP+FP+FN}$
- AUC(Area under the ROC Curve):
  - ROC:Receiver Operating Characteristics
  - TPR(True Positive Rate): $\frac{TP}{TP+FN}$
  - FPR(False positive rate): $\frac{FP}{TN+FP}$
  - [introduction](https://blog.csdn.net/u013385925/article/details/80385873)
- logloss: $y -(\log(p) + (1-y) \log(1-p))$

## Encoding

- binarization
- label encoding
- one hot encoding
- embedding

## Binary Regression Metrics

- AE(absolute error): $|y_{true} - y_{pred}|$
- MAE(mean absolute error): $\frac{1}{N} \sum_{i=1}^N |y_{true_i} - y_{pred_i}|$
- SE(squared error): $(y_{true} - y_{pred})^2$
- MSE(mean squared error): $\frac{1}{N} \sum_{i=1}^N (y_{true_i} - y_{pred_i})^2$
- SLE(squared log error): $(\log(1+y_{true}) - \log(1+y_{pred}))^2$
- MSLE(mean squared log error): $\frac{1}{N} \sum_{i=1}^N (\log(1+y_{true_i}) - \log(1+y_{pred_i}))^2$
- RMSE(root MSE): $\sqrt{MSE}$
- RMSLE(root MSLE): $\sqrt{MSLE}$
- $R^2$ (coefficient of determination): 
  - $R^2 = 1 - \frac{\sum_{i=1}^N (y_{true_i} - y_{pred_i})^2}{\sum_{i=1}^N (y_{true_i} - mean)^2}$
  - $mean = \frac{i}{N} \sum_{i=1}^N y_{true_i}$
