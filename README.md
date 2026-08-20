# Heavy Equipment Selling Price Prediction

## Executive Summary
This project addresses the challenge of predicting realized selling prices (in USD) for industrial heavy equipment across global auction transactions. Valuations in the heavy machinery industry—encompassing bulldozers, hydraulic excavators, wheel loaders, cranes, and forklifts—are influenced by complex interactions between asset age, usage intensity, technical specifications, geographic demand, and market timing.

Developed as a Machine Learning Practice (MLP) project, this repository contains an end-to-end regression pipeline designed to predict equipment prices while minimizing the Root Mean Squared Logarithmic Error (RMSLE). The pipeline features a strict leakage-free architecture, out-of-fold (OOF) cross-validation target encoding, domain-specific feature engineering, and a tuned LightGBM gradient boosting model that achieves an RMSLE of **0.1973** on validation data (a substantial improvement over the baseline RMSLE of 0.2263).

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Architecture](#dataset-architecture)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering and Preprocessing](#feature-engineering-and-preprocessing)
- [Model Architecture and Comparison](#model-architecture-and-comparison)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Final Submission Pipeline](#final-submission-pipeline)
- [Project Setup and Execution](#project-setup-and-execution)
- [Technology Stack](#technology-stack)

---

## Problem Statement

### Objective
Given transactional, technical, and operational metadata for a piece of heavy equipment at the time of sale, predict its final realized transaction price (`TargetValue` in USD).

### Problem Type and Metric
- **ML Task:** Supervised Non-linear Regression
- **Target Variable:** `TargetValue` (Continuous USD price ranging from $7,500 to $142,000)
- **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE)

$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2}$$

RMSLE penalizes relative ratio errors rather than absolute dollar deviations. An error of $5,000 on a $10,000 forklift incurs a higher penalty than the same dollar error on a $100,000 crane. Consequently, modeling `log1p(TargetValue)` is the natural formulation for training gradient boosted decision trees.

---

## Dataset Architecture

| Property | Detail |
|---|---|
| **Training Records** | 138,701 transactions |
| **Test Records** | 15,000 transactions |
| **Raw Features** | 49 (Operational, technical, geographic, vendor, and temporal attributes) |
| **Target Variable** | `TargetValue` (Realized auction price in USD) |
| **Machine Base Classes** | 1,249 unique equipment classes |
| **Vendors & Regions** | 31 vendor partner IDs across 53 regional codes |
| **Date Range** | January 31, 1990 to April 28, 2013 |
| **Missing Data Profile** | Significant sparsity — 40.8% missing operational hours; up to 99.9% missing in specialized spec columns |

---

## Exploratory Data Analysis

1. **Target Distribution and Log Transformation:**
   - The raw target distribution exhibits strong right-skewness (skewness coefficient: **0.97**), with a heavy concentration of transactions between $10,000 and $40,000 and a long upper tail reaching $142,000.
   - Applying logarithmic transformation `np.log1p(TargetValue)` normalizes the target distribution (skewness reduced to **-0.20**), stabilizing variance and directly aligning optimization with the RMSLE metric.

2. **Feature Sparsity:**
   - Columns `col18` and `col19` contained over 99.9% missing values (only 66 non-null entries out of 138,701 rows) and were dropped to prevent noise injection.
   - Operational meter hours (`OperationalHoursMeter`) were missing in 40.8% of entries, necessitating relative usage features (`HoursPerYear` and class-normalized usage indices).

3. **Categorical High Cardinality:**
   - Features such as `ProductConfigID` (3,594 unique values), `Spec_BaseClass` (1,249 unique values), `RegionCode` (53 values), and `VendorPartnerID` (31 values) required specialized out-of-fold encoding to prevent memory explosion from standard One-Hot Encoding.

---

## Feature Engineering and Preprocessing

The preprocessing pipeline strictly avoids data leakage by ensuring all fit operations (frequency computation, target encoding means, class-level statistics) are derived solely from training folds and applied downstream to validation and test splits.

```
+-------------------------------------------------------------------------------+
|                             RAW TRANSACTION DATA                              |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                            LEAK-FREE SPLIT (80/20)                             |
|                    Train: 110,960 rows | Val: 27,741 rows                    |
+-------------------------------------------------------------------------------+
                                       |
             +-------------------------+-------------------------+
             |                                                   |
             v                                                   v
+------------------------------------------+   +-----------------------------------+
|            TEMPORAL & DOMAIN             |   |        SPEC & TEXT FEATURES       |
| - AssetAge = SaleYear - ManufactureYear  |   | - Spec Descriptor String Length   |
| - HoursPerYear = Usage / (Age + 1)       |   | - Numeric extraction from specs   |
| - Age Grouping ('New', 'MidLife', etc.)  |   | - Interaction features:           |
| - Cyclical Month & Quarter Sin/Cos       |   |   Base_SubClass, Inventory_Func   |
+------------------------------------------+   +-----------------------------------+
             |                                                   |
             +-------------------------+-------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                    OUT-OF-FOLD TARGET ENCODING (5-FOLD CV)                    |
| High-cardinality features (RegionCode, ProductConfigID, Spec_BaseClass, etc.)|
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                  FREQUENCY ENCODING & LOW-CARDINALITY OHE                     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                SENTINEL IMPUTATION (-999) & MATRIX ALIGNMENT                  |
+-------------------------------------------------------------------------------+
```

### Key Transformations
1. **Temporal & Usage Indicators:**
   - `AssetAge`: Transaction year minus manufacturing year (clipped at non-negative values).
   - `HoursPerYear`: Annualized meter hours `OperationalHoursMeter / (AssetAge + 1)`.
   - `hours_vs_class_mean`: Ratio of equipment operational hours to its `Spec_BaseClass` mean.
   - `AgeGroup`: Categorical bucketed asset lifespan (`New`, `MidLife`, `Old`, `VeryOld`).
   - Cyclical encodings: Sine and cosine transformations for months and quarters.

2. **Composite Interactions:**
   - Combined high-signal categorical pairs: `Base_SubClass`, `Base_Drivetrain`, and `Inventory_Function`.

3. **Out-of-Fold Target Encoding (5-Fold CV):**
   - High-cardinality variables (`RegionCode`, `ProductConfigID`, `VendorPartnerID`, `Spec_BaseClass`, `Spec_SubClass`, `FunctionalClassification`, `DataOriginCode`, `AssetID`, etc.) were target-encoded using 5-fold cross-validation on the training set.
   - Validation and test splits were mapped using training global means as fallback to prevent target leakage.

4. **One-Hot & Frequency Encoding:**
   - Frequency of category occurrence mapped onto all sets.
   - Low-cardinality categories one-hot encoded; columns aligned across train, validation, and test sets.

---

## Model Architecture and Comparison

Three baseline ensemble algorithms were evaluated on an 80/20 train/validation split using `log1p(TargetValue)` as the optimization target:

| Model Architecture | Implementation | Validation RMSLE | Notes |
|---|---|---|---|
| **HistGradientBoosting** | `sklearn.ensemble.HistGradientBoostingRegressor` | 0.226326 | Native histogram-based boosting baseline |
| **LightGBM (Default)** | `lightgbm.LGBMRegressor` | 0.226345 | Default parameters baseline |
| **XGBoost (Default)** | `xgboost.XGBRegressor` | 0.221562 | Default histogram-tree baseline |
| **Tuned LightGBM** | `lightgbm.LGBMRegressor` (Optimized) | **0.197349** | **Tuned hyperparameters + Early Stopping** |

---

## Hyperparameter Optimization

Hyperparameters for LightGBM were systematically tuned to maximize tree capacity while maintaining regularization against overfitting:

```python
best_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 255,
    "min_child_samples": 15,
    "colsample_bytree": 0.7,
    "subsample": 0.9,
    "subsample_freq": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "random_state": 42,
    "verbose": -1
}
```

### Tuning Impact
- Decreasing the learning rate to `0.01` and expanding tree capacity (`num_leaves=255`) allowed the model to capture deep interactions among target-encoded features.
- Early stopping triggered at ~4,800 iterations, yielding an RMSLE score of **0.197349** (a **12.8% relative error reduction** over baseline models).

---

## Feature Importance Analysis

Feature importance analysis reveals that target-encoded regional and equipment classification features dominate the model's split decisions:

```
+----------------------------------+------------------+
| Feature                          | Importance Score |
+----------------------------------+------------------+
| RegionCode_TE                    | 90,854           |
| ProductConfigID_TE               | 63,062           |
| VendorPartnerID_TE               | 58,022           |
| Spec_BaseClass_TE                | 55,755           |
| FunctionalClassification_TE     | 53,874           |
| DataOriginCode_TE                | 46,966           |
| Spec_SubClass_TE                 | 40,353           |
| AssetID_TE                       | 27,852           |
| Spec_VariantModifier_TE          | 21,081           |
| Spec_ReleaseSeries_TE            | 14,352           |
+----------------------------------+------------------+
```

---

## Final Submission Pipeline

1. **Full Training Retraining:**
   - The tuned LightGBM model configuration was retrained on 100% of the training dataset (138,701 transactions) for `best_iteration_` rounds.
2. **Test Matrix Alignment:**
   - `X_test` features were reindexed to guarantee exact column alignment with `X_train`, filling missing test dummy columns with 0.
3. **Inverse Log Transformation:**
   - Predictions were transformed back to standard USD scale via `np.expm1(y_pred)` and clipped at 0:
     $$\hat{y}_{\text{final}} = \max(0, e^{\hat{y}_{\text{log}}} - 1)$$
4. **Output Generation:**
   - Formatted predictions written to `submission.csv` (`TransactionID`, `TargetValue`).

---

## Project Setup and Execution

### Environment Requirements
- Python 3.9+
- Anaconda / Jupyter Notebook environment

### Dependencies Installation
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn lightgbm xgboost
```

### Running the Notebook
1. Clone or place the project files into your working directory:
   ```bash
   git clone https://github.com/SathyaH02/SkillFusion.git
   cd SkillFusion
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "24f2005474-notebook-2026t2 (1).ipynb"
   ```
3. Execute all cells sequentially. The notebook will run EDA, build features, train models, output validation metrics, and save `submission.csv`.

---

## Technology Stack

- **Language:** Python 3.9
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical Tools:** `scipy.stats`
- **Machine Learning Frameworks:** `scikit-learn`, `lightgbm`, `xgboost`
- **Development Environment:** Jupyter Notebook / Kaggle Kernel

---

## License & Attribution
This repository is maintained as part of the Machine Learning Practice (MLP) coursework. Dataset provided via Heavy Equipment Selling Price Prediction Challenge.
