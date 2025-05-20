import streamlit as st
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")  # or just "nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    words = [word for word in words if word.isalnum()]

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message below to check if it's likely to be spam or not.")

# Input field
input_sms = st.text_area("✉️ Your SMS Message", height=150)

# Predict button
if st.button('🔍 Detect Spam'):
    if not input_sms.strip():
        st.warning("Please enter a message before clicking Detect.")
    else:
        # Preprocess, vectorize, predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0][prediction]

        # Display result
        if prediction == 1:
            st.error(f"🚨 This message is likely **Spam** (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ This message is **Not Spam** (Confidence: {probability:.2%})")

        # Optionally show transformed input
        with st.expander("🔍 See processed text"):
            st.code(transformed_sms)
