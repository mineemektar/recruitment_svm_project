# model.py
#
# This module contains functions for training and evaluating an SVM classification model.
# 
# Functions:
# - train_svm: Trains a Support Vector Machine (SVM) classifier using the specified parameters.
# - evaluate_model: Evaluates the trained model using accuracy and classification report.
#                   Uses a custom threshold (default 0.3) to convert probabilities into binary predictions.
#
# Example usage:
# model = train_svm(X_train, y_train)
# accuracy, report = evaluate_model(model, X_test, y_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def train_svm(X_train, y_train, C=1.0, kernel='linear', gamma='scale'):
    model = SVC(probability=True,C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict_proba(X_test)[:,1] > 0.3).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

