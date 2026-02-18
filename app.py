import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load model
model = pickle.load(open("model/email_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(
        w for w in text.split()
        if w not in stop_words
    )

def predict_email(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    category = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    return category, confidence

email = input("Enter email text: ")

category, confidence = predict_email(email)

print("\nPrediction:", category)
print("Confidence:", round(confidence * 100, 2), "%")
