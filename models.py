import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, List, Union, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

class EmailClassifier:
    def __init__(self, model_path: str = None):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
        else:
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    min_df=2,
                    max_df=0.85,
                    preprocessor=self.preprocess_text
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ))
            ])

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)

    def train(self, data: pd.DataFrame, text_column: str, label_column: str) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            data[text_column], data[label_column],
            test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "metrics": report,
            "accuracy": report.get("accuracy", 0),
            "test_samples": len(X_test)
        }

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]

    def save_model(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str) -> Pipeline:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def train_on_real_data(self, dataset_path: str) -> None:
        df = pd.read_csv("combined_emails_with_natural_pii.csv")
        df = df.rename(columns={"email": "email_text", "type": "category"})
        df.dropna(subset=["email_text", "category"], inplace=True)
        self.train(df, "email_text", "category")

