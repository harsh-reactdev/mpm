# Random Forest Approach for Predictive Maintenance

## Overview
Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time. It is a robust bagging-based approach that is less prone to overfitting than a single decision tree.

## Key Features in this Implementation
- **Class Balance**: We use `class_weight='balanced'` to ensure the model doesn't ignore the rare failure events.
- **Bagging Strategy**: By averaging multiple trees, the model reduces variance and provides stable predictions.
- **Robustness**: Random Forest handles outliers and noisy sensor data well without requiring strict scaling.

## Why use Random Forest?
- **Ease of Use**: Requires very little hyperparameter tuning to get good results.
- **No Scaling Required**: Unlike linear models, it doesn't care if temperatures are in hundreds and torque is in double digits.
- **Feature Interaction**: Naturally captures interactions between features (like temperature and pressure).

Check the `results/` folder for metrics and visualization.
