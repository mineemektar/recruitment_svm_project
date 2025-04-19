# app.py
#
# This FastAPI application serves a trained SVM model to make predictions
# based on `experience_years` and `technical_score` inputs.
#
# Usage:
# Send a POST request to `/predict` with JSON data:
# {
#     "experience_years": 2.5,
#     "technical_score": 80
# }
# Response:
# {
#     "prediction": 0
# }


from pydantic import BaseModel
import joblib
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

# Load the trained SVM model
model = joblib.load('models/svm_model.pkl')


# Define request body schema
class Data(BaseModel):
    experience_years: float
    technical_score: int


# Prediction endpoint
@app.post("/predict")
def predict(data: Data):
    try:
        # Prepare input features
        features = [data.experience_years, data.technical_score]

        # Make prediction
        prediction = model.predict([features])

        # Return prediction result
        return {"prediction": int(prediction[0])}

    except Exception as e:
        # Return error message in case of failure
        return {"error": f"An error occurred while predicting: {str(e)}"}
