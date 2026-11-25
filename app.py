import streamlit as st
import pandas as pd
import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩"
)

@st.cache_resource
def train_model():
    os.makedirs("data", exist_ok=True)

    zip_path = "data/sms_spam.zip"
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(
                f,
                sep="\t",
                names=["label", "message"]
            )

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, len(df)


st.title("📩 Email & SMS Spam Detection")
st.write("Machine Learning Based Spam Classification System")

with st.spinner("Training machine learning model..."):
    model, accuracy, total_messages = train_model()

col1, col2 = st.columns(2)

col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Dataset Size", total_messages)

st.divider()

st.subheader("Test Your Message")

message = st.text_area(
    "Enter your message:",
    placeholder="Type your message here..."
)

if st.button("Check Message"):
    if message.strip():

        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is LEGITIMATE!")

    else:
        st.warning("Please enter a message.")

st.caption("Algorithm: TF-IDF + Multinomial Naive Bayes")
st.caption("Dataset: UCI SMS Spam Collection")