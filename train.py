import pandas as pd
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(
        w for w in text.split()
        if w not in stop_words
    )

df = pd.read_csv("data/emails.csv")

df['cleaned'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))


print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

pickle.dump(model, open("model/email_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
