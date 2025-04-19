import pandas as pd
from src.preprocessing import preprocess_data, split_train_test, split_features_target
from src.model import train_svm, evaluate_model

# Veriyi yükle ve ön işle
df = preprocess_data('../data/dataset.csv')

# Özellikler (X) ve hedef değişkeni (y) ayır
X, y = split_features_target(df, 'target_column')

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = split_train_test(X, y)

# Modeli eğit
model = train_svm(X_train, y_train)

# Modeli değerlendir
accuracy, report = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy}")
print(report)
