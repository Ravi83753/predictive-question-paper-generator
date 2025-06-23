import pandas as pd
import numpy as np

def generate_question_paper(df, model, vectorizer, top_k=10):
    X = vectorizer.transform(df["question"])
    predictions = model.predict(X)
    df["score"] = predictions
    top_questions = df.sort_values(by="score", ascending=False).head(top_k)
    return top_questions[["question", "score"]]
