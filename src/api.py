from pydantic import BaseModel
import joblib
from fastapi import FastAPI

# FastAPI uygulaması
app = FastAPI()

# Modeli yükle
model = joblib.load('models/svm_model.pkl')


class Data(BaseModel):
    experience_years: float
    technical_score: int


@app.post("/predict")
def predict(data: Data):
    try:
        # Sadece 2 özelliği kullanarak tahmin yapıyoruz
        features = [data.experience_years, data.technical_score]

        # Tahmin yap
        prediction = model.predict([features])

        # Sonucu döndür
        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": f"An error occurred while predicting: {str(e)}"}
