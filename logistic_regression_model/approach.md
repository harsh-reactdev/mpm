# Logistic Regression Approach for Predictive Maintenance

## Overview
Logistic Regression is a fundamental linear classification algorithm. While simpler than tree-based models, it provides a clear baseline and is highly interpretable.

## Key Features in this Implementation
- **Feature Scaling**: Unlike tree-based models, Logistic Regression is sensitive to the scale of input features. We use `StandardScaler` to ensure all sensors (K, rpm, Nm) are on the same local scale.
- **Class Weighting**: We use `class_weight='balanced'` to handle the fact that healthy states far outnumber failure states.
- **Interpretabiltiy**: The coefficients of the model can be directly related to the probability of failure.

## Why use Logistic Regression?
- **Speed**: Extremely fast training and inference.
- **Baseline**: Useful to see if a simple linear boundary can separate failures from normal operation.
- **Probability Estimation**: Provides well-calibrated probabilities of the output class.

## How to Run
```bash
../venv/bin/python train_lr.py
```
Check the `results/` folder for metrics and the confusion matrix.
