from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset.csv"
MODEL = ROOT / "complaint_classifier.joblib"

df = pd.read_csv(DATA).dropna(subset=["complaint_text", "category"])
X_train, X_test, y_train, y_test = train_test_split(df.complaint_text, df.category, test_size=0.2, stratify=df.category, random_state=42)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, zero_division=0))
joblib.dump(model, MODEL)
print(f"Saved model: {MODEL}")
