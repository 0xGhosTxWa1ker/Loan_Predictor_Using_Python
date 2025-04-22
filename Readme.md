This project implements a 'Loan Prediction System' using various supervised machine learning algorithms. It predicts loan approval status based on user information such as income, credit history, education level, employment, and more.

## 📘 Overview

Loan eligibility prediction helps financial institutions quickly assess the risk of approving a loan application. This notebook takes a data-driven approach to predict loan approval by exploring the data, preprocessing it, training multiple classifiers, and evaluating their performance.

## 🔍 Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Encoding of categorical variables
- Model training using:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting Classifier
- Hyperparameter tuning with `GridSearchCV` for optimal performance
- Model evaluation with accuracy scores and confusion matrices

## ⚙️ Tech Stack

- Python 3
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## 📊 Model Performance

Each model is evaluated based on accuracy and confusion matrix.

Hyperparameter tuning using `GridSearchCV` is applied to optimize:
- `n_estimators`, `max_depth`, `criterion` for tree-based models

## 🔮 Future Work

- Integrate model into a web application (Flask, Streamlit)
- Use cross-validation with multiple folds for better generalization
- Add Explainable AI (SHAP, LIME) for interpretability
- Test on real-time user input

## 📚 Dataset

Dataset used is commonly available on platforms like [Kaggle](https://www.kaggle.com/datasets) under titles such as "Loan Prediction Dataset." Place the dataset file (`loan.csv`) in the notebook directory before running.
