# Customer Happiness Prediction

## Project Overview

This project develops a binary classification model to predict customer happiness for a logistics and on-demand delivery company using customer survey responses.

The objective is to predict whether a customer is happy (1) or unhappy (0) based on their satisfaction with different aspects of the delivery experience.

An additional objective is to identify the smallest set of informative features that preserves predictive performance, potentially reducing the number of questions required in future customer surveys.

## Dataset

The dataset, `ACME-HappinessSurvey2020.csv`, contains 126 customer observations.

The target variable is:

- **Y:** Customer happiness (0 = Unhappy, 1 = Happy).

The six original ordinal predictors measure customer satisfaction across different aspects of the delivery experience:

| Feature | Description |
|---|---|
| X1 | Order delivered on time |
| X2 | Contents of order matched expectations |
| X3 | Customer ordered everything they wanted |
| X4 | Customer paid a good price |
| X5 | Customer satisfaction with courier |
| X6 | Ease of ordering through the app |

Each predictor takes an integer value between 1 and 5, with higher values indicating greater satisfaction.

The dataset represents the publicly provided training observations. The remaining observations are reserved by the organizers as a private test set.

## Feature Engineering

To capture overall satisfaction patterns beyond individual survey responses, 18 aggregate features were engineered.

For each original question \(X_i\), three features were calculated using the other five survey responses, excluding \(X_i\):

1. **Other-question count of 5s:** Number of remaining questions with a response equal to 5.
2. **Other-question count of 4+:** Number of remaining questions with a response greater than or equal to 4.
3. **Other-question count of 2 or below:** Number of remaining questions with a response less than or equal to 2.

This produces 18 aggregate predictors in total.

The purpose of these features is to capture broader satisfaction patterns across the other survey dimensions while retaining information specific to each excluded question.

## Modelling Approach

A **Random Forest Classifier** was selected as the final modelling algorithm.

Random Forest combines predictions from multiple decision trees and can capture nonlinear relationships and interactions between predictors without requiring them to be explicitly specified.

The modelling process incorporates feature selection and hyperparameter optimization.

### 1. Feature Selection: RFECV

Recursive Feature Elimination with Cross-Validation (RFECV) is used to identify a reduced subset of predictive features.

Starting with all 18 engineered features, RFECV progressively eliminates less informative predictors. At each stage, model performance is evaluated using stratified cross-validation, with classification accuracy used as the selection criterion.

The number of retained features is determined by the cross-validation results.

### 2. Hyperparameter Optimization

After selecting the feature subset, RandomizedSearchCV is used to optimize the Random Forest hyperparameters.

The search evaluates different combinations of:

- Number of trees (`n_estimators`).
- Maximum tree depth (`max_depth`).
- Minimum samples required to split a node (`min_samples_split`).
- Minimum samples required in a leaf (`min_samples_leaf`).
- Number of features considered at each split (`max_features`).
- Class weighting (`class_weight`).

The configuration with the highest mean cross-validation accuracy is selected.

### 3. Nested Cross-Validation

A nested cross-validation framework is used to estimate predictive performance while accounting for feature selection and hyperparameter optimization.

The procedure follows two levels:

- **Outer cross-validation:** Stratified 7-fold cross-validation provides an independent evaluation of the complete modelling procedure.
- **Inner cross-validation:** Within each outer training fold, RFECV first selects the features, followed by RandomizedSearchCV to optimize the Random Forest hyperparameters using the selected features.

The outer validation fold is not used during feature selection or hyperparameter tuning.

This provides a more reliable estimate of generalization performance than reporting the best cross-validation score from the hyperparameter search alone.

### 4. Final Model Training

After evaluating the modelling procedure, feature selection is repeated using all available labelled observations. Hyperparameter optimization is then performed on the selected features using stratified cross-validation.

The final Random Forest is subsequently trained using all 126 available observations and the selected features.

## Final Selected Features

The final feature selection procedure retained four predictors from the original 18 engineered features:

| Selected Feature | Description |
|---|---|
| `X2_other_count4plus` | Number of other survey questions with responses of 4 or 5, excluding X2 |
| `X4_other_count5` | Number of other survey questions with responses equal to 5, excluding X4 |
| `X5_other_count4plus` | Number of other survey questions with responses of 4 or 5, excluding X5 |
| `X6_other_count4plus` | Number of other survey questions with responses of 4 or 5, excluding X6 |

These features collectively capture broader satisfaction patterns across different combinations of survey dimensions.

Their selection suggests that aggregate satisfaction measures can provide useful predictive information beyond individual survey responses.

## Model Evaluation

Classification accuracy is the primary evaluation metric:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

The following performance measures are examined:

- Accuracy across outer cross-validation folds.
- Mean nested cross-validation accuracy.
- Standard deviation of cross-validation accuracy.
- Overall out-of-fold accuracy.
- Final model training accuracy.

Training accuracy is reported separately as a diagnostic measure and is not treated as an estimate of generalization performance.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib

## Project Objective

The primary objective is to achieve at least 73% classification accuracy on the organizers' private test dataset.

Beyond predictive performance, the project investigates whether aggregate satisfaction measures can reduce the number of required survey questions while preserving useful predictive information.

The ultimate goal is to develop a predictive model that helps identify unhappy customers and provides insights into the factors associated with customer satisfaction.
