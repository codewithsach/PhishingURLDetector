import streamlit as st
import joblib

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Phishing URL Detector")
st.write("Enter a URL and the model will predict if it is **phishing** or **legitimate**.")

# Load saved model + vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("phishing_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

url = st.text_input("Enter URL", placeholder="e.g., https://www.google.com")

col1, col2 = st.columns(2)

with col1:
    check = st.button("Check URL")

with col2:
    clear = st.button("Clear")

if clear:
    st.rerun()

if check:
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        features = vectorizer.transform([url])
        pred = model.predict(features)[0]

        if pred == 1:
            st.error("⚠️ Phishing URL Detected")
        else:
            st.success("✅ Legitimate URL")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression (trained on phishing URL dataset)")
