# main.py
#
# This script serves as the entry point for training the SVM model and setting up the FastAPI application.
# It performs the following steps:
# 1. Loads and preprocesses the dataset using the preprocess_data function.
# 2. Trains the SVM model using the train_svm function.
# 3. Evaluates the model's performance with the evaluate_model function.
# 4. Saves the trained model as a .pkl file using joblib.
# 5. Starts a FastAPI application to serve the model via an API endpoint.
#
# Main functions:
# - preprocess_data: Loads and preprocesses the dataset.
# - train_svm: Trains the SVM model using the provided training data.
# - evaluate_model: Evaluates the model's accuracy and generates a classification report.
# - uvicorn.run: Starts the FastAPI application to expose the trained model for predictions via a POST request.
#
# Example usage:
# Run the script, and it will:
# - Train the model.
# - Save the model to 'models/svm_model.pkl'.
# - Start the FastAPI application at http://127.0.0.1:8000.
#

import pandas as pd
from src.preprocessing import preprocess_data
from src.model import train_svm, evaluate_model
import joblib
import uvicorn


def main():
    # Load the data and preprocess it
    X_train, X_test, y_train, y_test = preprocess_data('data/dataset.csv')

    # Train the SVM model
    model = train_svm(X_train, y_train, C=1.0, kernel='linear', gamma='scale')

    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Save the model
    model_path = 'models/svm_model.pkl'
    joblib.dump(model, model_path)

    # Import FastAPI app after the model is saved
    from src.api import app

    # Start the FastAPI application
    print("Starting the API...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
