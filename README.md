
# 🧠 İşe Alım Tahmin Modeli (SVM + SMOTE)

Bu proje, adayların işe alınıp alınmayacağını **`experience_years`** ve **`technical_score`** verilerine dayanarak tahmin eden bir makine öğrenmesi uygulamasıdır. Dengesiz veri problemini çözmek için **SMOTE (Synthetic Minority Over-sampling Technique)** yöntemi kullanılmış ve ardından **Support Vector Machine (SVM)** algoritması ile model eğitilmiştir.

---

## 📌 Özellikler

- 🔍 Dengesiz sınıf problemi çözümü (SMOTE)
- 🤖 SVM modeli ile sınıflandırma
- ⚖️ Olasılık temelli tahmin ve özel eşik değeri (threshold = 0.3)
- 📊 Performans değerlendirmesi (`classification_report`)
- 📁 Basit proje yapısı

---

## 📁 Proje Yapısı

```
recruitment_svm_project/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── svm_model_development.py
│
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── api.py (FastAPI dosyası)
│
├── models/                # Create this directory for model storage
│   └── svm_model.pkl      # Model will be saved here
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Kurulum

1. Reposu klonlayın:
   ```bash
   git clone https://github.com/kullanici-adi/proje-adi.git
   cd proje-adi
   ```

2. Sanal ortam oluşturun ve etkinleştirin:
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   ```

3. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔬 Kullanım

### ✅ 1. Veriyi yükle & böl
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv")
X = df[["experience_years", "technical_score"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```

### 🔁 2. SMOTE ile dengele
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

### 🧠 3. Modeli eğit
```python
from model import train_svm

model = train_svm(X_train_res, y_train_res)
```

### 📈 4. Değerlendir
```python
from model import evaluate_model

accuracy, report = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy)
print(report)
```

---

## 🛠 Kullanılan Kütüphaneler

```txt
scikit-learn
pandas
numpy
imblearn
```

`pip freeze > requirements.txt` komutuyla güncellenebilir.

---

## 📌 Notlar

- Sınıf dağılımı SMOTE ile eşitlenmiştir:
  ```
  0    7057
  1    7057
  ```
- SVC modelinde `probability=True` sayesinde `predict_proba` kullanılabilir.
- `threshold=0.3` ile daha esnek ve hataya toleranslı kararlar alınabilir.

---