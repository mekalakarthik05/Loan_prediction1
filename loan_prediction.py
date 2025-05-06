import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('loan_prediction.csv')

imputer = SimpleImputer(strategy='most_frequent')
data['LoanAmount'] = imputer.fit_transform(data[['LoanAmount']])
data['Credit_History'] = imputer.fit_transform(data[['Credit_History']])
data['Loan_Amount_Term'] = imputer.fit_transform(data[['Loan_Amount_Term']])

label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

y = data['Loan_Status']

# Use more informative features
X = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
          'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
          'Credit_History', 'Property_Area']]

# Feature scaling (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with enough components to retain ~95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

inputs = {}
inputs['Gender'] = int(input("Enter Gender (1=Male, 0=Female): "))
inputs['Married'] = int(input("Enter Married (1=Yes, 0=No): "))
inputs['Dependents'] = int(input("Enter Dependents (0/1/2/3): "))
inputs['Education'] = int(input("Enter Education (1=Graduate, 0=Not Graduate): "))
inputs['Self_Employed'] = int(input("Enter Self Employed (1=Yes, 0=No): "))
inputs['ApplicantIncome'] = float(input("Enter Applicant Income: "))
inputs['CoapplicantIncome'] = float(input("Enter Coapplicant Income: "))
inputs['LoanAmount'] = float(input("Enter Loan Amount: "))
inputs['Loan_Amount_Term'] = float(input("Enter Loan Amount Term (e.g., 360): "))
inputs['Credit_History'] = float(input("Enter Credit History (1 or 0): "))
inputs['Property_Area'] = int(input("Enter Property Area (0=Rural, 1=Semiurban, 2=Urban): "))

new_data = pd.DataFrame([inputs])
new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)

prediction = model.predict(new_data_pca)
if prediction == 1:
    print("Loan Sanctioned")
else:
    print("Loan Not Sanctioned")
