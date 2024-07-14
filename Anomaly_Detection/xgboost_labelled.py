import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('feature_list.csv')

# Separate features (X) and labels (y)
X = df.drop(columns=['file', 'type'])
y = df.iloc[:, -1]   # Last column (labels)

# Convert labels to binary format if necessary ('OK' -> 0, 'Anomaly' -> 1)
y = y.apply(lambda x: 0 if x == 'OK' else 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = XGBClassifier(
    objective='binary:logistic',   # Binary classification with logistic regression
    eval_metric='logloss',         # Metric to evaluate during training
    learning_rate=0.1,             # Learning rate (default is 0.3)
    n_estimators=100,              # Number of boosting rounds (default is 100)
    max_depth=15,                   # Maximum depth of trees (default is 6)
    subsample=0.8,                 # Fraction of samples used for training each tree (default is 1.0)
    colsample_bytree=0.9,          # Fraction of features used for training each tree (default is 1.0)
    random_state=42                # Random state for reproducibility
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
