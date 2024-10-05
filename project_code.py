import pandas as pd

pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# metadata
print(bank_marketing.metadata)

# variable information
print(bank_marketing.variables)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

# Convert features and target to a DataFrame for better inspection
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

# Print the first few rows of the dataset
print(X_df.head())
print(y_df.head())

# Print the column names
print(X_df.columns)
print(y_df.columns)

# Define features (adjust these names if they do not match exactly)
feature_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing',
                   'loan', 'contact',  'month', 'duration', 'campaign',
                   'pdays', 'previous', 'poutcome']

# Since the target might be named differently, ensure it matches correctly
target_column = 'y'  # This might be 'purchase' or something similar

# Assign features and target variables
X = X_df[feature_columns]
y = y_df[target_column]

# Encode categorical variables
X = pd.get_dummies(X)

# Check if the target needs encoding (assuming binary classification)
if y.dtype == 'object':
    y = pd.get_dummies(y, drop_first=True)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = model.feature_importances_
features_list = X.columns

print("Feature Importance:")
for feature, importance in zip(features_list, feature_importance):
    print(f"{feature}: {importance:.4f}")

# Plotting the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=features_list, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()

# Predict probabilities for the test set
y_prob = model.predict_proba(X_test)[:, 1]

# Get the probability for a specific customer (example: first customer in the test set)
customer_index = 1  # Index of the customer in the test set
customer_probability = y_prob[customer_index]
print(f"Probability that the customer will take the service: {customer_probability:.2f}")
