# Credit Risk Analysis Report

## Overview of the Analysis

In this analysis, we evaluated the performance of a logistic regression model for predicting loan default risks. The dataset contains financial information on loans, including their status as either healthy (0) or high-risk (1). The objective was to predict whether a loan is high-risk based on various features.

The key variables in the dataset included loan status (`loan_status`) as the target variable, and several financial and personal attributes as features. We performed the following stages in the machine learning process:
- Data preparation: Loaded and cleaned the data, and separated it into features and labels.
- Model training: Used logistic regression to fit the model on the training data.
- Model evaluation: Assessed the model's performance using metrics such as accuracy, precision, and recall.

## Code:
**1) Split the Data into Training and Testing Sets**
'# Read the CSV file into a Pandas DataFrame'
df = pd.read_csv('/Users/mauriciogil/Desktop/credit-risk-classification/credit-risk-classification/Credit_Risk/lending_data.csv')

'# Review the DataFrame'
df.head()

**2) Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.**
'# Separate the y variable, the labels (loan_status column)'
y = df['loan_status']

'# Separate the X variable, the features (all columns except loan_status)'
X = df.drop(columns='loan_status')

**3) Step 3: Split the data into training and testing datasets by using train_test_split.**
'# Import the train_test_split module
from sklearn.model_selection import train_test_split

'# Split the data into training and testing datasets with a random_state of 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

'# Review the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

**4) Create a Logistic Regression Model with the Original Data**
'# Import the LogisticRegression module from sklearn
from sklearn.linear_model import LogisticRegression

'# Instantiate the Logistic Regression model with a random_state of 1
logreg_model = LogisticRegression(random_state=1)

'# Fit the model using the training data (X_train and y_train)
logreg_model.fit(X_train, y_train)

**5) Step 3: Evaluate the model’s performance by doing the following:**
-Generate a confusion matrix.
-Print the classification report.
from sklearn.metrics import confusion_matrix, classification_report

'# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

'# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

## Results

* Logistic Regression Model:
    * **Accuracy Score:** 0.99
    * **Precision Score for Class 0 (Healthy Loan):** 1.00
    * **Recall Score for Class 0 (Healthy Loan):** 1.00
    * **Precision Score for Class 1 (High-Risk Loan):** 0.86
    * **Recall Score for Class 1 (High-Risk Loan):** 0.91

## Summary

The logistic regression model demonstrates strong performance with an overall accuracy of 0.99. It achieves perfect precision and recall for healthy loans (`0`), and a high precision of 0.86 and recall of 0.91 for high-risk loans (`1`). This indicates that the model effectively identifies both loan statuses, though it is slightly less effective at predicting high-risk loans compared to healthy loans. 

Given the high performance metrics, the model is recommended for use by the company, particularly for identifying high-risk loans where the recall is crucial. If further improvements are needed, additional features or model tuning could be explored to enhance the prediction of high-risk loans.

### Credits: XPert Learning Assistant, Stack Overflow, Class Recordings/Materials
