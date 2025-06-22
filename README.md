# 🛸 Titanic - Spaceship ML Competition

This repository contains my solution for the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic), 
where the goal is to predict whether passengers were transported to an alternate dimension after a mysterious event aboard the Spaceship Titanic.

This is a **binary classification** problem using **tabular data**, which includes a mix of categorical and numerical features.

---

## 📊 Project Overview

This notebook walks through the complete machine learning pipeline:
- **Data exploration (EDA)** with statistical summaries and visual insights
- **Feature engineering**:
  - Created new features such as `TotalSpent`, `RoomNum`, `Deck`, and `Side`
  - Handled missing values using tailored imputation strategies
  - Removed irelevant features (such as unique 'Name' feature)
- **Data preprocessing**:
  - Scaled numerical features
  - Encoded categorical variables
- **Model training**: Compared multiple classifiers and applied hyperparameter tuning
- **Evaluation**: Assessed performance on the train and validation sets using accuracy, confusion matrix and several other metrices before submission
- **Interpretability**: Used SHAP values and feature importance plots to explain model decisions

---

## 🧰 Tools & Libraries
- Python 3  
- pandas, numpy  
- seaborn, matplotlib  
- scikit-learn  
- SHAP

---

## 🏆 Model Performance

- **Training Accuracy**: ~0.82  
- **Validation Accuracy**: 0.8160  
- **Kaggle Public Leaderboard Score**: 0.80406  
- **Leaderboard Placement**: Top 20%

The current best-performing model is a Histogram-based Gradient Boosting Classifier (HGB) from scikit-learn. It outperformed other models, including XGBoost, based on the validation score achieved with the selected hyperparameter configurations.

These results show **consistent performance** across training, validation, and public leaderboard sets, suggesting that **overfitting is not a primary issue**. 
The small gap between validation and test scores reflects a generally stable model.

---

### 🧠 Insights & Next Steps

Since increasing the dataset size isn't an option and overfitting is under control, future improvement efforts should focus on:

- **Advanced Feature Engineering**:  
  - Create feature interactions (e.g., ratios, combined behavior signals)
  - Apply domain-inspired logic to craft abstract features (e.g., travel group proxies)
  - Use insights from model interpretability (e.g., SHAP) to guide new feature creation
  - Consider binning skewed numerical features to reduce variance or increase stability

- **Smarter Encoding for Categorical Variables**:  
  - Explore **target encoding**, **frequency encoding**, or **leave-one-out encoding** for variables like `HomePlanet`, `Destination`, and `Cabin`

- **Model Optimization & Ensembling**:  
  - Increase sample size or iteration depth in `RandomizedSearchCV`
  - Experiment with **Optuna** for more efficient hyperparameter optimization
  - Try blending XGBoost and HGB predictions or stacking with a meta-learner

- **Handling Missing Data Strategically**:  
  - Explore whether missingness itself is informative (Missing Not At Random)
  - Add binary indicators for missing fields
  - Use **model-based imputation** techniques

- **Error Analysis**:  
  - Examine false positives and false negatives for patterns
  - Apply SHAP locally to understand individual misclassifications

- **Other Possibilities**:  
  - Experiment with deep learning models such as **TabNet** or **MLP**
  - Apply feature selection techniques to reduce noise and dimensionality
  - Submit and blend multiple model versions for ensemble comparisons

---

## 🔗 Link to the Kaggle competition

[https://www.kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic)
