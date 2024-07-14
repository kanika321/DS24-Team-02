import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb

# Load CSV
df = pd.read_csv('feature_list1.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['file', 'type'])  
y = df['type']  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LightGBM classifier
clf = lgb.LGBMClassifier(
    boosting_type='gbdt',       
    objective='binary',         
    metric='binary_error',      
    learning_rate=0.1,          
    n_estimators=100,           
    max_depth=5,                 
    num_leaves=31,              
    random_state=42             
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


