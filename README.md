🏦 Loan Prediction Case Study using PCA and Random Forest
This project predicts loan approval status based on applicant details using machine learning techniques. It applies Principal Component Analysis (PCA) for dimensionality reduction and uses a Random Forest Classifier to achieve high prediction accuracy.

📁 Project Overview
The goal is to determine whether a loan should be sanctioned based on various applicant features like income, credit history, education, etc. The model is trained on historical data and can also accept real-time user inputs for prediction.

📊 Dataset
The dataset loan_prediction.csv contains:

Categorical and numerical fields related to applicant demographics and financials.

The target column: Loan_Status (1 = Approved, 0 = Not Approved).

🧰 Tools & Technologies
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Label Encoding

Imputation

Scaling

PCA

Random Forest Classifier

🧠 Workflow Summary
Data Preprocessing

Missing values handled using SimpleImputer (most frequent strategy).

Categorical columns encoded using LabelEncoder.

Feature Engineering

Features selected for training include:

Gender, Marital Status, Dependents, Education, Self Employed, Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area.

Features scaled using StandardScaler.

PCA applied to retain 95% of variance while reducing dimensionality.

Model Training

Random Forest Classifier with 200 estimators and max depth of 8.

Data split into 80% training and 20% testing sets.

Evaluation

Accuracy Score

Classification Report

Confusion Matrix (visualized using heatmap)

User Prediction

Model accepts user input for new loan applications.

Predicts and prints whether the loan will be sanctioned.

📈 Model Performance
Accuracy: ~82–86%

Classifier Used: RandomForestClassifier(n_estimators=200, max_depth=8)

PCA applied to improve feature quality and model generalization.

