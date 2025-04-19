# svm_model_development.py

#This script loads and preprocesses the dataset, splits it into training and test sets,
# trains an SVM classifier, and evaluates its performance.

import pandas as pd
from src.preprocessing import preprocess_data, split_train_test, split_features_target
from src.model import train_svm, evaluate_model

# Load and preprocess the dataset
df = preprocess_data('../data/dataset.csv')

# Split into features (X) and target (y)
X, y = split_features_target(df, 'target_column')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_train_test(X, y)

# Train the SVM model
model = train_svm(X_train, y_train)

# Evaluate the model
accuracy, report = evaluate_model(model, X_test, y_test)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(report)

