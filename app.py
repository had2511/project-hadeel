import streamlit as st
import pickle

# Load model and vectorizer
with open('news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Streamlit App UI
st.title("📰 News Article Fake/Real Classifier")
st.markdown("Enter a news article and click **Check** to see if it's **FAKE** or **REAL**.")

user_input = st.text_area("Paste the news article content here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess and predict
        text = [user_input]
        vector = tfidf.transform(text).toarray()
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("🚨 This news article is likely **FAKE**.")
        else:
            st.success("✅ This news article is likely **REAL**.")
