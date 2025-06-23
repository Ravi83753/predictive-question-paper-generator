import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df, text_column="question"):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    return X, vectorizer, df
