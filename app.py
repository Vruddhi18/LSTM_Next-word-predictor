import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import gdown

# Download model and tokenizer files
url_model = "https://drive.google.com/uc?id=1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG"
output_model = "next_words.keras"
output_tokenizer = "token.pkl"

gdown.download(url_model, output_model, quiet=False)
gdown.download("https://drive.google.com/uc?id=1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG", output_tokenizer, quiet=False)

# Load model and tokenizer
model = load_model(output_model)

with open(output_tokenizer, 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict the next word
def predict_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence), axis=-1)
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    return predicted_word

# Streamlit UI
st.title("Next Word Predictor")
st.write("Enter some text, and I'll predict the next word for you!")

input_text = st.text_input("Enter text:", "")
if st.button("Predict"):
    if input_text:
        text = input_text.split(" ")
        text = " ".join(text[-3:])  # Use last 3 words for prediction
        predicted_word = predict_word(model, tokenizer, text)
        st.success(f"The predicted next word is: **{predicted_word}**")
    else:
        st.warning("Please enter some text.")
