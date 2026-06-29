# 🏦 LoanTap – Personal Loan Underwriting with Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Binary Classification | Credit Risk Modeling | Precision-Recall Tradeoff**

---

## 📌 Problem Statement

LoanTap is an online lending platform delivering customized loan products to millennials. The data science team needs to build an **underwriting layer** to determine the creditworthiness of individual borrowers for **Personal Loans**.

**Goal:** Given a set of borrower attributes, predict whether a credit line should be extended — and if so, recommend appropriate repayment terms.

- `1` → **Charged Off** (defaulter — do NOT lend)
- `0` → **Fully Paid** (creditworthy — safe to lend)

---

## 📂 Repository Structure

```
loantap-loan-underwriting/
│
├── LoanTap_Case_Study.ipynb     # Complete analysis notebook
├── https://drive.google.com/file/d/1ZPYj7CZCfxntE8p2Lze_4QO4MyEOy6_d/view?usp=sharing      # Dataset (396,030 records × 27 features)
├── README.md                    # You are here
│
└── plots/                       # All generated visualizations
    ├── 01_target_distribution.png
    ├── 02_numerical_univariate.png
    ├── 03_categorical_univariate.png
    ├── 04_loan_status_by_grade.png
    ├── 05_int_rate_by_status.png
    ├── 06_correlation_heatmap.png
    ├── 07_loan_amnt_vs_installment.png
    ├── 08_home_ownership.png
    ├── 09_top_emp_titles.png
    ├── 10_coefficients.png
    ├── 11_roc_auc_curve.png
    ├── 12_precision_recall_curve.png
    ├── 13_precision_recall_threshold.png
    ├── 14_confusion_matrix.png
    └── 15_confusion_matrix_0_3.png
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | LoanTap Internal Data |
| Records | 3,96,030 |
| Features | 27 |
| Target Variable | `loan_status` (Fully Paid / Charged Off) |
| Class Distribution | 80.4% Fully Paid · 19.6% Charged Off |

### Key Features

| Feature | Description |
|---|---|
| `loan_amnt` | Listed loan amount applied for |
| `int_rate` | Interest rate on the loan |
| `grade` / `sub_grade` | LoanTap assigned risk grade |
| `emp_length` | Employment length (0–10+ years) |
| `annual_inc` | Self-reported annual income |
| `dti` | Debt-to-income ratio |
| `revol_util` | Revolving credit utilization rate |
| `term` | Loan term — 36 or 60 months |
| `purpose` | Borrower's stated loan purpose |
| `mort_acc` | Number of mortgage accounts |
| `pub_rec` | Number of derogatory public records |

---

## 🔬 Approach

```
Data Loading → EDA → Preprocessing → Feature Engineering → Model Building → Evaluation → Insights
```

### 1. Exploratory Data Analysis
- Shape, dtypes, missing value audit
- Univariate distributions (numerical + categorical)
- Bivariate analysis — loan status vs grade, interest rate, DTI, home ownership
- Correlation heatmap — identified `loan_amnt` ↔ `installment` correlation of **~0.95**

### 2. Data Preprocessing
- Dropped irrelevant columns (`emp_title`, `title`, `address`, `issue_d`, `earliest_cr_line`)
- Missing value treatment:
  - `emp_length` → filled with `'Unknown'`
  - `revol_util`, `mort_acc` → filled with median
  - `pub_rec_bankruptcies` → filled with `0`
- Outlier capping at 1st–99th percentile for `annual_inc`, `revol_bal`, `dti`

### 3. Feature Engineering
- Binary flags (as per case study guidelines):
  - `pub_rec_flag` — any derogatory public record
  - `mort_acc_flag` — any mortgage account
  - `pub_rec_bankruptcies_flag` — any bankruptcy
- Ordinal encoding for `grade`, `sub_grade`, `emp_length`
- One-hot encoding for `home_ownership`, `verification_status`, `purpose`, etc.

### 4. Modeling
- **Algorithm:** Logistic Regression (`sklearn`)
- **Scaling:** MinMaxScaler
- **Class Imbalance:** handled via `class_weight='balanced'`
- **Split:** 80% train / 20% test (stratified)

---

## 📈 Results

### Model Performance

| Metric | Threshold = 0.5 | Threshold = 0.3 (NPA-Safe) |
|---|---|---|
| ROC AUC | **0.7101** | 0.7101 |
| Recall (Charged Off) | 0.63 | **0.93** |
| Precision (Charged Off) | 0.32 | 0.23 |
| F1 (Charged Off) | 0.43 | 0.37 |

### ROC AUC Curve
![ROC AUC](plots/11_roc_auc_curve.png)

### Precision-Recall Curve
![PR Curve](plots/12_precision_recall_curve.png)

### Precision & Recall vs Threshold
![Threshold Analysis](plots/13_precision_recall_threshold.png)

### Confusion Matrices
| Default Threshold (0.5) | NPA-Safe Threshold (0.3) |
|---|---|
| ![CM 0.5](plots/14_confusion_matrix.png) | ![CM 0.3](plots/15_confusion_matrix_0_3.png) |

### Top Feature Coefficients
![Coefficients](plots/10_coefficients.png)

---

## 💡 Key Insights

1. **Interest Rate** is a strong default predictor — Charged Off loans carry significantly higher rates
2. **Grade A borrowers** repay ~94% of loans vs ~66% for Grade G (Q4: TRUE)
3. **Revolving utilization > 80%** is a red flag — borrowers maxing credit lines default more
4. **60-month loans** carry higher default risk than 36-month loans
5. **Joint applications** are safer — co-borrower acts as a risk buffer
6. **Debt consolidation** is the most common loan purpose but not the riskiest
7. **Top 2 job titles:** Teacher and Manager (highest loan applicant count)

---

## 🎯 Business Recommendations

| Priority | Recommendation | Rationale |
|---|---|---|
| 🔴 High | Use threshold **0.35** instead of default 0.5 | Better NPA control without excessive rejections |
| 🔴 High | Auto-reject sub-grade **F/G** borrowers | Default rates exceed 35% |
| 🟡 Medium | Cap loan amount for **Grade C–D** | Limit exposure on mid-risk segment |
| 🟡 Medium | Prefer **36-month** term offers | Lower default rates vs 60-month |
| 🟢 Low | Prioritize **Joint applications** | Statistically lower default risk |
| 🟢 Low | Flag **revol_util > 80%** for manual review | Strong signal for financial stress |

### Why Recall over Precision?

From a bank's perspective, **missing a real defaulter (False Negative) is far more costly** than rejecting a creditworthy borrower (False Positive). Every missed defaulter = an NPA on the books. Hence:

> **Primary metric → Recall for the "Charged Off" class**

A lower threshold (0.3) catches **93% of real defaulters** at the cost of some good-loan rejections — which is the acceptable tradeoff to stay NPA-safe.

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualizations |
| `scikit-learn` | Logistic Regression, scaling, metrics |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/loantap-loan-underwriting.git
cd loantap-loan-underwriting

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook LoanTap_Case_Study.ipynb
```

> **Note:** The dataset file `logistic_regression.csv` must be in the same directory as the notebook.

---

## 📋 Questionnaire Answers

| # | Question | Answer |
|---|---|---|
| Q1 | % of customers who fully paid | **80.39%** |
| Q2 | Correlation: Loan Amount & Installment | **~0.95** (very high positive — higher loan = higher EMI) |
| Q3 | Majority home ownership | **MORTGAGE** |
| Q4 | Grade A more likely to fully pay? | **TRUE** (~94% repayment rate) |
| Q5 | Top 2 job titles | **Teacher, Manager** |
| Q6 | Primary metric for bank | **Recall** (minimise missed defaulters) |
| Q7 | Precision-Recall gap impact | Low recall at 0.5 → 37% of defaulters slip through → NPA risk |
| Q8 | Top features affecting outcome | `sub_grade`, `revol_util`, `int_rate`, `dti`, `annual_inc`, `term` |
| Q9 | Results affected by geography? | **YES** — state-level economic conditions impact default rates |

---

## 👤 Author

**Krishna Agarwal**
Data Analyst · M.S. Data Science (Scaler Neoversity)
[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)

---

*Case study submitted as part of M.S. Data Science curriculum at Scaler Neoversity.*
