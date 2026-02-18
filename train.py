import pandas as pd
import nltk
import re
import pickle
import kagglehub
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(w for w in text.split() if w not in stop_words)

path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")

files = os.listdir(path)
csv_file = [f for f in files if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(path, csv_file), encoding="latin-1")

df = df[['text', 'spam']]

df['label'] = df['spam'].map({
    0: "Not Spam",
    1: "Spam"
})

df['cleaned'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=3000,
    max_df=0.9
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline = 1 / len(df['label'].unique())
print("\nRandom Baseline Accuracy:", baseline)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

pickle.dump(model, open("model/email_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
