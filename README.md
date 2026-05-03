# Credit Card Fraud Detection

A machine learning pipeline built to classify fraudulent credit card transactions using Python and scikit-learn.

---

## Overview

This project trains and evaluates two classification models — Logistic Regression and Random Forest — on a real-world dataset of 284,807 credit card transactions, of which only 492 (0.17%) are fraudulent. The core challenge is handling severe class imbalance while maximizing fraud detection recall.

---

## Results

| Model | Recall (Fraud) | ROC-AUC |
|---|---|---|
| Logistic Regression | ~76% | ~0.97 |
| Random Forest | ~86% | ~0.98 |

Random Forest outperformed Logistic Regression on recall — the most important metric for fraud detection, where missing actual fraud is more costly than a false alarm.

---

## Tech Stack

- **Python 3.10+**
- **pandas** — data loading and manipulation
- **scikit-learn** — model training, scaling, evaluation
- **imbalanced-learn** — RandomOverSampler for class imbalance
- **matplotlib / seaborn** — confusion matrix and feature importance visualization

---

## Dataset

**Credit Card Fraud Detection** — publicly available on Kaggle  
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions, September 2013 (European cardholders)
- Features V1–V28 are PCA-transformed for anonymization
- `Amount` = transaction amount, `Class` = 0 (normal) / 1 (fraud)

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn
```

**3. Download the dataset**  
Download `creditcard.csv` from Kaggle (link above) and place it in the project root folder.

**4. Run the notebook**
```bash
jupyter notebook
```
Open `fraud_detection.ipynb` and run all cells.

---

## Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.ipynb   # Main notebook — full pipeline
├── README.md               # This file
└── creditcard.csv          # Dataset (download from Kaggle, not included in repo)
```

---

## Key Steps

1. **Load & explore data** — understand the class distribution
2. **Visualize imbalance** — see how rare fraud cases actually are
3. **Oversample with RandomOverSampler** — balance training data before modeling
4. **Scale features** — StandardScaler on Amount and Time columns
5. **Train models** — Logistic Regression vs. Random Forest
6. **Evaluate** — classification report, confusion matrix, ROC-AUC score
7. **Feature importance** — identify which variables most predict fraud

---

## Credits & References

Tutorial reference: [ML Credit Card Fraud Detection — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/)

Dataset: [ULB Machine Learning Group — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Author

**Duy Nguyen**  
Fort Worth, TX | Finance & Technology  
GitHub: [github.com/DuyBienNguyen](https://github.com/DuyBienNguyen)
