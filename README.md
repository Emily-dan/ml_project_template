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
- Precision: `$\frac{TP}{TP+FP}$`
- Recall: `$\frac{TP}{TP+FN}$`
- F1-score(F1): `$\frac{2RP}{R+P}=\frac{2TP}{2TP+FP+FN}$`
- AUC(Area under the ROC Curve):
  - ROC:Receiver Operating Characteristics
  - TPR(True Positive Rate): `$\frac{TP}{TP+FN}$`
  - FPR(False positive rate): `$\frac{FP}{TN+FP}$`
  - [introduction](https://blog.csdn.net/u013385925/article/details/80385873)
- logloss: `$y -(\log(p) + (1-y) \log(1-p))$`

## Encoding

- binarization
= label encoding
- one hot encoding
