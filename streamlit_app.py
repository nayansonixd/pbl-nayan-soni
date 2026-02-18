import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = stopwords.words('english')

model = pickle.load(open("model/email_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(
        w for w in text.split()
        if w not in stop_words
    )

st.title("📧 Spam Detection Dashboard")
st.write("Enter email text to classify as Spam or Not Spam.")

if "history" not in st.session_state:
    st.session_state.history = []

email_input = st.text_area("Email Text")

if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter email text.")
    else:
        cleaned = clean_text(email_input)
        vector = vectorizer.transform([cleaned])
        category = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])

        st.session_state.history.append({
            "Email": email_input[:50],
            "Category": category,
            "Confidence (%)": round(confidence * 100, 2)
        })

        if category == "Spam":
            st.error(f"⚠ Spam (Confidence: {round(confidence*100,2)}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {round(confidence*100,2)}%)")

# ----------------------------
# Analytics
# ----------------------------
if st.session_state.history:
    st.subheader("📊 Analytics")

    df = pd.DataFrame(st.session_state.history)

    st.write("### Category Distribution")
    fig, ax = plt.subplots()
    df["Category"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.write("### Prediction History")
    st.dataframe(df)
