# main.py

import pandas as pd
from src.preprocessing import preprocess_data
from src.model import train_svm, evaluate_model
import joblib
import uvicorn


def main():
    # Veriyi yükle ve işleme adımlarını gerçekleştir
    X_train, X_test, y_train, y_test = preprocess_data('data/dataset.csv')

    # Modeli eğit
    model = train_svm(X_train, y_train, C=1.0, kernel='linear', gamma='scale')

    # Modeli değerlendir
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Modeli kaydet
    model_path = 'models/svm_model.pkl'
    joblib.dump(model, model_path)

    # Model kaydedildikten sonra api import edilsin
    from src.api import app

    # FastAPI uygulamasını başlat
    print("API'yi başlatıyorum...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
