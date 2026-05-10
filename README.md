# Credit Card Fraud Detection

Binary classification project for detecting fraudulent credit card transactions using machine learning.

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days
- 492 fraud cases (0.17%) — heavily imbalanced
- Features: V1–V28 (PCA-transformed), Amount, Time, Class

## Project Structure

```
Credit_Card_Fraud_Detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   └── Model.ipynb
├── requirements.txt
└── README.md
```

## Approach

### EDA
- Class distribution analysis — confirmed 0.17% fraud rate
- Descriptive statistics via `df.describe()`
- Null check — no missing values

### Preprocessing
- Train/test split: 80/20 with `stratify=y` to preserve class ratio
- `Amount` scaled with `StandardScaler` (fit on train only, transform on test)
- `Time` converted to `Hour` (seconds → hour of day via `% 24`) — fraud activity may correlate with time of day
- V1–V28 left as-is (already PCA-transformed)

### Imbalance Handling
Two strategies compared:
- **class_weight / scale_pos_weight** — penalizes the model for missing fraud without generating synthetic data
- **SMOTE** — generates synthetic fraud samples to balance classes (applied only on train set)

### Models
Trained sequentially from simple to complex:

| Model | Imbalance strategy |
|---|---|
| Logistic Regression | baseline (no handling) |
| Logistic Regression | SMOTE |
| LightGBM | class_weight='balanced' |
| LightGBM | SMOTE |
| XGBoost | scale_pos_weight (GridSearchCV) |
| XGBoost | SMOTE (GridSearchCV) |

## Results

| Model | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|
| LogReg baseline | 0.653 | 0.831 | 0.731 | 0.954 |
| LogReg + SMOTE | 0.908 | 0.054 | 0.102 | 0.972 |
| LightGBM + class_weight | 0.837 | 0.812 | 0.824 | 0.964 |
| LightGBM + SMOTE | 0.867 | 0.491 | 0.627 | 0.970 |
| **XGBoost + scale_pos_weight** | **0.827** | **0.880** | **0.853** | **0.975** |
| XGBoost + SMOTE | 0.857 | 0.785 | 0.820 | 0.981 |

**Best model: XGBoost with scale_pos_weight** — F1=0.853, AUC-ROC=0.975

Best parameters found via GridSearchCV:
```
learning_rate: 0.2
max_depth: 5
n_estimators: 200
scale_pos_weight: 577.3
```

### Key findings
- SMOTE consistently increased Recall but hurt Precision — more false alarms
- `scale_pos_weight` in XGBoost gave the best balance between catching fraud and avoiding false positives
- Accuracy is not a useful metric here — a model predicting everything as "not fraud" would score 99.83%

## Tech Stack

```
Python, pandas, NumPy, scikit-learn
XGBoost, LightGBM, imbalanced-learn
Matplotlib, Seaborn
```

## Installation

```bash
pip install -r requirements.txt
```

## Author

Dilmurat — [github.com/Dilmurat090](https://github.com/Dilmurat090)
