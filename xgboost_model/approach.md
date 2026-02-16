# XGBoost Approach for Predictive Maintenance

## Overview
XGBoost (Extreme Gradient Boosting) is a powerful implementation of gradient boosted decision trees. It is particularly well-suited for this predictive maintenance task due to its ability to handle imbalanced data and capture non-linear sensor relationships.

## Key Features in this Implementation
- **Scale Position Weight**: Since only ~3% of the samples are failures, we use `scale_pos_weight` to help the model pay more attention to the minority (failure) class.
- **Engineered Interactions**: We included `Power` and `Temperature_Difference` which our initial analysis showed were high contributors.
- **Regularization**: XGBoost includes built-in L1 and L2 regularization to prevent overfitting on the sensor noise.

## Why use XGBoost?
- **Imbalance Handling**: Superior to many other algorithms when the "failure" event is rare.
- **Accuracy**: Often provides the highest predictive performance for tabular data.
- **Feature Importance**: Inherently provides ranking of sensors that contribute most to the decision.

## How to Run
```bash
../venv/bin/python train_xgboost.py
```
Check the `results/` folder for the confusion matrix and metrics.
