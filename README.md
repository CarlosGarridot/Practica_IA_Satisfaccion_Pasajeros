# Airline Passenger Satisfaction Prediction
**Artificial Intelligence**
Carlos Garrido del Toro & Elena del Pilar Fernández Wyzynska
January 2026

---

## Description

Supervised machine learning project that predicts whether an airline passenger will be **satisfied or dissatisfied** with their flight experience, based on demographics, trip characteristics, and service ratings (wifi, food, comfort, etc.).

Five classification models are compared and evaluated across multiple metrics, including class bias, overfitting, and computational efficiency.

---

## Dataset

**Airline Passenger Satisfaction Dataset**
Source: [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- 25,976 rows × 25 columns
- Target variable: `satisfaction` (Satisfied / Neutral or Dissatisfied)

---

## Project Pipeline

### 1. Preprocessing
- Removal of irrelevant columns (`id`, `Unnamed`)
- Missing-value handling: mean imputation for numeric variables
- Categorical encoding with `LabelEncoder`
- Normalization with `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Unsupervised clustering with **K-Means** to uncover passenger profiles
- `k=3` selected via the **Elbow Method** — the inflection point falls at `k=2`, but `k=3` was chosen for finer-grained passenger segments without losing meaningful structure
- Identified groups: demanding business travelers vs. relaxed tourists, characterized by age, flight distance, class, and travel type

### 3. Train/Test Split & Evaluation Metrics
- Stratified 70/30 split (`stratify=y`) to preserve class balance in both sets
- Evaluation metrics defined up front: Accuracy, Confusion Matrix, Precision, Recall, and F1-score — chosen specifically to catch class imbalance issues that raw accuracy alone would hide

### 4. Supervised Models
Five classification algorithms, each tuned via `GridSearchCV` with 5-fold cross-validation:
- **Perceptron** — baseline linear model
- **Logistic Regression** — tuned over `C`, `solver`, and `max_iter`; feature importance shows Online boarding, Seat Comfort, and Inflight entertainment as the strongest positive drivers, Departure Delay and personal travel type as the strongest negative ones
- **SVM** — tuned over `C`, `kernel`, and `gamma`; RBF kernel (94.32%) clearly outperforms linear (87.4%), confirming the satisfaction/service relationship is non-linear, with an optimal `C=1.0`
- **Decision Tree** — tuned via GridSearch to the best single-tree configuration
- **Random Forest** — bagging ensemble of deep, unpruned trees to reduce the variance a single tree suffers from

### 5. Comparative Analysis
- **Class bias**: separate F1-score and Recall for satisfied vs. dissatisfied customers, per model — e.g. the Perceptron shows a clear negative bias (Recall 0.78 for "satisfied" vs. 0.86 for "dissatisfied")
- **Error types**: False Positives vs. False Negatives from each model's confusion matrix, with business interpretation (a false positive — predicting "satisfied" for an actually dissatisfied customer — is read as churn risk)
- **Overfitting**: train vs. test accuracy gap for every model (Random Forest: 100% train / 95.7% test, a 4.3-point gap accepted as the cost of capturing non-linear structure the linear models underfit)
- **Efficiency**: training time and per-sample prediction latency

## Key Findings

- Tree-based models (Random Forest, Decision Tree) clear 94% accuracy; linear models (Logistic Regression, Perceptron) plateau at 82–87% — clear evidence that passenger satisfaction depends on non-linear interactions a hyperplane can't separate.
- Random Forest reaches a perfect 100% training score but only a 4.3-point gap to its 95.7% test score — GridSearch's own selection, trading a small, controlled amount of overfitting for a +1.3-point accuracy gain over a single decision tree.
- SVM is the closest runner-up (95.11% test accuracy, 2.68-point gap) with the best generalization-to-performance balance of all five models, at a higher computational cost than Random Forest.

## Conclusion

**Random Forest** is the winning model: highest accuracy, fewest False Positives (139 cases, 1.78% churn risk), and balanced behavior across classes. SVM is the runner-up, with very similar performance but higher computational cost.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)

---

## How to Run

```bash
# 1. Clone the repository
# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
# 3. Download the dataset from Kaggle and place it in the project root as:
#    airline_passenger_satisfaction.csv
# 4. Open the notebook
jupyter notebook airline_satisfaction.ipynb
```

---
