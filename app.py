import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()                         
    text = re.sub(r'[^a-z\s]', '', text)    
    return " ".join(
        w for w in text.split()             
        if w not in stop_words              
    )

model = pickle.load(open("model/email_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_email(text):
    cleaned = clean_text(text)                 
    vector = vectorizer.transform([cleaned])   
    category = model.predict(vector)[0]        


    priority_map = {
        "Urgent": 5,
        "Follow-up": 3,
        "Informational": 2,
        "Spam": 1
    }

    priority = priority_map.get(category, 0)

    return category, priority

email = input("Enter email text: ")

category, priority = predict_email(email)

print("\nPredicted Category:", category)
print("Priority Score:", priority)
