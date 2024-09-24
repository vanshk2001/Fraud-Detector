import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
fraud_dataset = pd.read_csv(r'C:\Users\dell\OneDrive\Documents\vansh folder\DSAI Course\Capstone Project\Fraud_Analysis_Dataset.csv')

# Data preprocessing
# Create Balance Diff. features
fraud_dataset['bal_diff_orig'] = fraud_dataset['newbalanceOrig'] - fraud_dataset['oldbalanceOrg']
fraud_dataset['bal_diff_dest'] = fraud_dataset['newbalanceDest'] - fraud_dataset['oldbalanceDest']

# Create Balance Diff. / Amount features
fraud_dataset['bal_diff_orig/amount'] = fraud_dataset['bal_diff_orig'] / fraud_dataset['amount']
fraud_dataset['bal_diff_dest/amount'] = fraud_dataset['bal_diff_dest'] / fraud_dataset['amount']

# Drop Features
fraud_dataset = fraud_dataset.drop(['step','nameOrig','nameDest'],axis=1)

# Applying One-Hot Encoding to Type column
fraud_dataset = pd.get_dummies(fraud_dataset,columns=['type'], drop_first=True, dtype='uint8')

# Defining the features variable and target variable
X = fraud_dataset.drop(['isFraud'],axis=1)
y = fraud_dataset['isFraud']

# Train the Random Forest model
model = RandomForestClassifier(max_depth=5, min_samples_leaf=2, n_estimators=50, random_state=42)
model.fit(X, y)

# Save the trained model to a file
dump(model, r'C:\Users\dell\OneDrive\Documents\vansh folder\DSAI Course\Capstone Project\fraud_detection_model.joblib')
