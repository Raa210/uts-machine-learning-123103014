"""LANGKAH 2 - Preprocessing, TF-IDF, training, dan evaluasi."""
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline

from preprocess import bersihkan

# 1. Baca data hasil crawling
df = pd.read_csv("data_ulasan.csv").dropna()

# 2. Labeling otomatis dari rating (rating 3 dibuang karena netral/ambigu)
df = df[df["rating"] != 3].copy()
df["label"] = df["rating"].apply(lambda r: "positif" if r >= 4 else "negatif")
print("Distribusi label:\n", df["label"].value_counts(), "\n")

# 3. Preprocessing teks
df["bersih"] = df["ulasan"].apply(bersihkan)
df = df[df["bersih"].str.len() > 0]

# 4. Split train/test 80:20 (stratify menjaga proporsi kelas)
X_train, X_test, y_train, y_test = train_test_split(
    df["bersih"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 5. Pipeline: TF-IDF -> Logistic Regression
# class_weight="balanced" menangani data yang tidak seimbang antar kelas
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(X_train, y_train)

# 6. Evaluasi
pred = model.predict(X_test)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

# 7. Simpan model untuk dipakai Streamlit
joblib.dump(model, "model.joblib")
print("Model tersimpan ke model.joblib")
