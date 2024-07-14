import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

#Load the CSV File
csv_path = r"../torch-condor-template/feature_list_combined.csv"
df = pd.read_csv(csv_path)

df = df.dropna()

print(df['type'].unique())

# Encode categorical variables
df = pd.get_dummies(df, columns=['type'], drop_first=False)

print(df.columns)

df['is_anomaly'] = df['type_NOK_Acoustic'] | df['type_NOK_Spindle_Irms']

# Drop the original type columns and the 'file' column
X = df.drop(columns=['file', 'type_NOK_Acoustic', 'type_NOK_Spindle_Irms', 'type_OK_Acoustic', 'type_OK_Spindle_Irms'])
y = df['is_anomaly']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost Model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3, reg_alpha=0.1, reg_lambda=0.1)
xgb_model.fit(X_train, y_train)

# Evaluate the Model
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

# Predict Anomalies on New Data
new_data = X_test.iloc[:5]  
predictions = xgb_model.predict(new_data)
print(f'Predictions for new data:\n{predictions}')
