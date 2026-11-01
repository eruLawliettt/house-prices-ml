# House Prices Regression

Machine learning project for the Kaggle **House Prices: Advanced Regression Techniques** competition.

The goal is to predict house sale prices based on structural, categorical and numerical property features.

## Environment

- Python 3.11.16

## Project Structure

```text
house-prices-ml/
├── config/
│   └── config.yaml
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_boosting_modeling.ipynb
│   ├── 04_dnn_modeling.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_tuning_models.ipynb
│   └── 07_ensembles.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── ensembles.py
├── main.py
├── requirements.txt
└── README.md
```

## Pipeline

The main pipeline includes:

- missing value handling based on feature semantics;
- categorical feature encoding;
- numerical feature scaling where required;
- native categorical processing for CatBoost;
- feature engineering;
- log transformation of the target;
- K-Fold cross-validation;
- training and comparison of multiple regression models;
- hyperparameter tuning;
- automatic selection of the best individual model;
- out-of-fold prediction generation;
- weighted model ensembling;
- final training and Kaggle submission generation.

## Target Transformation

The target variable is `SalePrice`.

Because house prices are strongly right-skewed, the project uses:

```python
y = np.log1p(SalePrice)
```

Models are trained and evaluated in log space.

Before generating the final Kaggle submission, predictions are converted back to the original price scale using:

```python
predictions = np.expm1(predictions)
```

## Preprocessing

Missing values are handled according to their semantic meaning.

Structural missing categorical values, such as missing garage, basement or pool information, are treated as the absence of the corresponding feature.

Examples:

```text
PoolQC
GarageType
GarageFinish
BsmtQual
BsmtExposure
FireplaceQu
```

These values are filled with:

```text
"None"
```

Structural numerical missing values are filled with `0`.

True missing values are imputed separately:

- categorical features → most frequent value;
- numerical features → median.

For standard models:

- categorical features are one-hot encoded;
- numerical features are standardized.

For boosting models:

- categorical features are one-hot encoded;
- numerical scaling is skipped.

For CatBoost:

- categorical features remain categorical;
- no one-hot encoding is applied;
- numerical scaling is skipped.

`MSSubClass` is treated as a categorical feature despite being stored numerically in the original dataset.

## Feature Engineering

Several engineered features were explored during validation.

The final pipeline includes:

```text
TotalSF
TotalBathrooms
HouseAge
```

### TotalSF

Total usable floor area:

```python
TotalSF = (
    TotalBsmtSF
    + 1stFlrSF
    + 2ndFlrSF
)
```

### TotalBathrooms

Weighted total number of bathrooms:

```python
TotalBathrooms = (
    FullBath
    + 0.5 * HalfBath
    + BsmtFullBath
    + 0.5 * BsmtHalfBath
)
```

### HouseAge

Age of the house at the time of sale:

```python
HouseAge = YrSold - YearBuilt
```

Feature engineering can be enabled or disabled through `config/config.yaml`.

## Models

The project includes:

```text
Elastic Net
Random Forest
LightGBM
XGBoost
CatBoost
Deep Neural Network
```

The DNN uses one-hot encoded tabular features and a fully connected architecture.

## Validation

Models are evaluated using **5-Fold K-Fold cross-validation** with shuffling.

The validation metric is Root Mean Squared Error in log space:

```text
RMSE(log1p(SalePrice))
```

The scikit-learn scoring configuration is:

```yaml
scoring: neg_root_mean_squared_error
```

The number of folds and random seed are controlled through `config/config.yaml`.

## Hyperparameter Tuning

Hyperparameters were tuned using Optuna.

The strongest tuned configurations included:

- CatBoost
- XGBoost
- LightGBM
- Random Forest
- Elastic Net
- DNN

The goal of tuning was not exhaustive leaderboard optimization, but to obtain strong and reproducible configurations for the final pipeline.

## Cross-Validation Results

Final tuned model results:

| Model | CV RMSE |
| --- | ---: |
| CatBoost | 0.1234 |
| XGBoost | 0.1270 |
| LightGBM | 0.1294 |
| Random Forest | 0.1392 |
| Elastic Net | 0.1408 |
| DNN | 0.1492 |

CatBoost achieved the strongest individual cross-validation score.

## Ensemble

The final ensemble combines:

```text
60% CatBoost
30% LightGBM
10% DNN
```

Out-of-fold predictions are generated for every model participating in the ensemble.

For each fold:

```text
train on 4 folds
↓
predict the remaining validation fold
↓
predict the Kaggle test set
```

This produces:

- OOF predictions for evaluating the ensemble without training leakage;
- test predictions from every fold, which are averaged before the final ensemble is created.

The weighted ensemble is calculated in log space:

```python
ensemble = (
    0.6 * catboost_predictions
    + 0.3 * lightgbm_predictions
    + 0.1 * dnn_predictions
)
```

The ensemble improved the OOF score compared with the individual models.

Example OOF results:

| Model | OOF RMSE |
| --- | ---: |
| CatBoost | 0.1257 |
| LightGBM | 0.1308 |
| DNN | 0.1546 |
| Equal-weight ensemble | 0.1262 |
| Weighted ensemble | 0.1235 |

## Kaggle Results

Final public leaderboard results:

| Submission | Public Score |
| --- | ---: |
| Weighted Ensemble | **0.12084** |
| CatBoost | 0.12325 |

The weighted ensemble improved the leaderboard score over the strongest individual model.

This demonstrates the benefit of combining models with different inductive biases even when some ensemble members are individually weaker.

## Configuration

The experiment is controlled through:

```text
config/config.yaml
```

Example:

```yaml
general:
  experiment_name: house_prices_final
  seed: 42
  use_ensemble_for_submission: true

target:
  column: SalePrice
  log_transform: true

cv:
  n_splits: 5
  scoring: neg_root_mean_squared_error

preprocessing:
  use_total_sf: true
  use_total_bathrooms: true
  use_house_age: true
```

Individual models can be enabled or disabled through the config.

Example:

```yaml
models:
  catboost:
    enabled: true

  lightgbm:
    enabled: true

  dnn:
    enabled: true
```

The final submission mode is controlled by:

```yaml
general:
  use_ensemble_for_submission: true
```

If set to `true`, the weighted ensemble is used for the final submission.

If set to `false`, the best individual model according to cross-validation is trained and used instead.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

The script:

1. loads the training dataset;
2. applies the configured target transformation;
3. evaluates all enabled models using cross-validation;
4. saves model results to `results.csv`;
5. identifies the best individual model;
6. loads the Kaggle test dataset;
7. either trains the best individual model or generates OOF predictions for the ensemble;
8. trains the required models using the configured preprocessing;
9. generates final test predictions;
10. converts predictions back from log space;
11. saves the Kaggle submission.

Depending on configuration, the final output is either:

```text
submission.csv
```

or:

```text
ensemble_submission.csv
```

## Notebooks

The `notebooks/` directory contains the complete experimental workflow:

- exploratory data analysis;
- missing-value analysis;
- preprocessing experiments;
- baseline regression models;
- boosting models;
- DNN experiments;
- feature engineering;
- hyperparameter tuning;
- OOF prediction generation;
- ensemble experiments;

## Final Result

The strongest individual model was CatBoost with a public leaderboard score of:

```text
0.12325
```

The final weighted ensemble improved this result to:

```text
0.12084
```

The project demonstrates a complete reproducible tabular regression workflow including preprocessing, feature engineering, model tuning, neural networks, cross-validation, out-of-fold predictions, model ensembling and Kaggle submission generation.