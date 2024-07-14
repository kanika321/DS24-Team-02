import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb

# Assuming you have a DataFrame 'df' with your data
# Replace this with your actual data loading process
# For example, reading from CSV
df = pd.read_csv('feature_list1.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['file', 'type'])  # Features
y = df['type']  # Target variable indicating anomaly or not

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LightGBM classifier
clf = lgb.LGBMClassifier(
    boosting_type='gbdt',        # Gradient Boosting Decision Tree
    objective='binary',          # Binary classification
    metric='binary_error',       # Metric to optimize during training
    learning_rate=0.1,           # Learning rate (default is 0.1)
    n_estimators=100,            # Number of trees (default is 100)
    max_depth=5,                 # Maximum depth of each tree (default is -1, no limit)
    num_leaves=31,               # Maximum number of leaves per tree (default is 31)
    random_state=42              # Random state for reproducibility
)

# Train the classifier
clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


