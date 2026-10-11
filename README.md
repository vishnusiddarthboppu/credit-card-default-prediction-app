# 💳 Credit Card Default Risk Predictor 
   🔗 **Live Demo:** https://credit-card-defaulter-prediction-app.streamlit.app/

A machine learning web app that predicts whether a credit card customer will **default on their payment next month**, using demographic details, repayment history, bill amounts and payment amounts. The model is an SVM classifier (SVC), and the app is built with Streamlit.

## Problem Statement

Financial institutions lose money when customers fail to repay their credit card dues. Identifying high-risk customers in advance helps banks reduce losses, assess credit risk better, and manage risky accounts proactively.

**Goal:** build a binary classification model that predicts whether a customer will default (`1`) or not (`0`).

## Dataset

- **Size:** 30,000 customers and 25 columns (ID, 23 input features, and the target)
- **Target:** `default` (`Y` = defaulter, `N` = not a defaulter), converted to `1` / `0`
- **Class balance:** about 77.9% not defaulters and 22.1% defaulters, so the data is imbalanced
- **Missing values:** none
- **Download:** [Google Drive link](https://drive.google.com/file/d/1xSYd7xZ5TAxYYG6_eP0ukCj97Hr9jE2O/view?usp=sharing) (a copy is also included in this repository)

### Features

| Group | Columns | Description |
|---|---|---|
| Demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Credit limit and customer details. `SEX` is F/M; `EDUCATION` has 6 categories; `MARRIAGE` has 4 categories |
| Repayment status | `PAY_0`, `PAY_2` to `PAY_6` | How late the payment was in each of the last 6 months |
| Bill amounts | `BILL_AMT1` to `BILL_AMT6` | Credit card statement amount (what is owed) for each of the last 6 months |
| Payment amounts | `PAY_AMT1` to `PAY_AMT6` | Amount actually paid in each of the last 6 months |

For every group of six columns, the number `1` is the most recent month and `6` is the oldest. `PAY_0` is the original dataset's name for what should have been `PAY_1`, so it was renamed to `pay_1` during training.

### Repayment status values

| Value | Meaning |
|---|---|
| `-2` | No credit use that month |
| `-1` | Paid in full |
| `0` | Paid the minimum only |
| `1` to `8` | Payment delayed by that many months |

Higher values mean worse repayment behavior and a higher risk of default.

## Approach

1. Dropped the `ID` column (it carries no predictive information).
2. Renamed `PAY_0` to `PAY_1` and converted all column names to lowercase.
3. Removed 35 duplicate rows, leaving 29,965 rows.
4. Converted the target `default` from `Y` / `N` to `1` / `0`.
5. Split the data 80% / 20% into training (23,972 rows) and test (5,993 rows) sets with `random_state=42`.
6. Built a scikit-learn pipeline:
   - `OrdinalEncoder` for `sex`, `education` and `marriage`
   - `RobustScaler` for the 20 numeric columns (the bill and payment amounts are heavily right-skewed with extreme values)
   - `SVC(kernel='rbf', gamma='scale', class_weight='balanced')`
7. Saved the whole pipeline with `pickle` as `default_trained_model.sav`, so the app can pass raw customer values straight to the model.

## Results (test set)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Not a defaulter) | 0.87 | 0.85 | 0.86 | 4,673 |
| 1 (Defaulter) | 0.52 | 0.55 | 0.53 | 1,320 |

**Overall accuracy:** 78.7%

**Confusion matrix**

| | Predicted: Not a defaulter | Predicted: Defaulter |
|---|---|---|
| **Actual: Not a defaulter** | 3,991 | 682 |
| **Actual: Defaulter** | 594 | 726 |

**Why SVC with balanced class weights?** Because about 78% of customers do not default, a model can look accurate while missing most defaulters. SVC with `class_weight='balanced'` gave the best F1-score for the defaulter class (0.53), ahead of the Naive Bayes (0.48) and Logistic Regression (0.35) results noted in the project notebook, so it was chosen as the final model.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Project Structure

```
├── default_prediction_web_app.py                 # Streamlit web app
├── default_predictive_system.py                  # Script to test a single prediction
├── default_trained_model.sav                     # Saved trained model (pipeline + SVC)
├── Credit_Card_Defaulter_Prediction_project.ipynb # Data analysis, preprocessing and model training
├── Credit_Card_Defaulter_Prediction.csv          # Dataset
├── requirements.txt                              # Python dependencies
├── .gitignore
└── README.md
```

## Run Locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run default_prediction_web_app.py
```

The app opens in your browser at `http://localhost:8501`.

> The saved model must be loaded with the same scikit-learn version it was trained with (see `requirements.txt`), otherwise loading may fail or give wrong results.

## Using the App

1. Enter the customer's credit limit, sex, education, marital status and age.
2. Select the repayment status for each of the last 6 months.
3. Enter the bill amounts and payment amounts for the last 6 months.
4. Click **Predict**. The app shows either *"The Person is a Defaulter"* or *"The Person is Not a Defaulter"*.

## Limitations

- Performance on the defaulter class is moderate (precision 0.52, recall 0.55), so the model both misses some defaulters and flags some good customers.
- The model was trained on one historical dataset and is **not** suitable for real lending decisions.
- Sex, education and marital status are used because they are in the dataset. In real credit scoring, using such attributes may be restricted by law and raises fairness concerns.

## Future Improvements

- Compare with gradient boosting models (XGBoost, LightGBM) using ROC-AUC
- Tune the SVC parameters (`C`, `gamma`) and the decision threshold
- Show a risk probability or score instead of only a yes/no result
- Add engineered features such as payment-to-bill ratios

## Disclaimer

This project is for learning and demonstration purposes only.

## Author

Vishn Siddarth Boppu
