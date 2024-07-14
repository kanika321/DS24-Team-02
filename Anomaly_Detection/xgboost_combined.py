import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# Step 1: Load the CSV File
csv_path = r"../torch-condor-template/feature_list_combined.csv"
df = pd.read_csv(csv_path)

# Step 2: Preprocess the Data
# Handle missing values if any
df = df.dropna()

print(df['type'].unique())

# Encode categorical variables if necessary (e.g., 'type')
df = pd.get_dummies(df, columns=['type'], drop_first=False)

print(df.columns)

# Separate features and labels
# Assuming 'type_NOK_Acoustic' and 'type_NOK_Spindle_Irms' columns indicate anomalies (1 for NOK, 0 for OK)
# We will create a new column 'is_anomaly' based on these
df['is_anomaly'] = df['type_NOK_Acoustic'] | df['type_NOK_Spindle_Irms']

# Drop the original type columns and the 'file' column
X = df.drop(columns=['file', 'type_NOK_Acoustic', 'type_NOK_Spindle_Irms', 'type_OK_Acoustic', 'type_OK_Spindle_Irms'])
y = df['is_anomaly']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the XGBoost Model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3, reg_alpha=0.1, reg_lambda=0.1)
xgb_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Calculate training accuracy
train_pred = xgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f'Training Accuracy: {train_accuracy}')

# Cross-Validation
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {scores}')

# Step 6: Predict Anomalies on New Data
new_data = X_test.iloc[:5]  # Example: Predict anomalies on the first 5 samples of the test set
predictions = xgb_model.predict(new_data)
print(f'Predictions for new data:\n{predictions}')
