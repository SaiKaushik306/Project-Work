# ✅ Drug Review Classification Project - Final Version (Random Forest Model)

# --------------------
# 1. Import Libraries
# --------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 2. Load and Clean the Dataset
# ----------------------------
data = pd.read_csv("C:\\Users\\saibh\\OneDrive\\Documents\\EXCELR\\PROJECT WORK\\drugsCom_raw.xlsx")  # adjust the path if needed

# Focus on 3 conditions
target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
data = data[data['condition'].isin(target_conditions)]

# Drop missing
data.dropna(subset=['review', 'rating', 'drugName'], inplace=True)

# ----------------------
# 3. Text Preprocessing
# ----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

data['clean_review'] = data['review'].apply(clean_text)

# Lemmatization with spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

data['clean_lemma_review'] = data['clean_review'].apply(lemmatize_text)

# ----------------------
# 4. Sentiment Analysis
# ----------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

data['sentiment_score'] = data['review'].apply(get_sentiment)

# -----------------------------
# 5. Feature Engineering (TF-IDF + Sentiment)
# -----------------------------
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(data['clean_lemma_review'])

# Scale sentiment score between 0 and 1
scaler = MinMaxScaler()
sentiment_scaled = scaler.fit_transform(data[['sentiment_score']])

# Combine features
X_final = hstack([X_tfidf, sentiment_scaled])

# --------------------
# 6. Encode Labels
# --------------------
le = LabelEncoder()
y_encoded = le.fit_transform(data['condition'])

# ----------------------------
# 7. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ----------------------------
# 8. Model: Random Forest
# ----------------------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# ----------------------------
# 9. Evaluation
# ----------------------------
print("\n✅ Accuracy (Random Forest):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------
# 10. Save Model & Vectorizer
# ----------------------------
joblib.dump(rf_model, "drug_condition_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "sentiment_scaler.pkl")
