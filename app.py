import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model('next_words.keras')
tokenizer = pickle.load(open('token.pkl', 'rb'))

# Prediction function
def predict_word(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence), axis=-1)
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    st.write(f"Predicted Next Word: **{predicted_word}**")
    return predicted_word

# Streamlit UI
st.title("Next Word Prediction")
st.markdown("Enter a sentence or phrase, and I will predict the next word.")

user_input = st.text_input("Type your text here:")

if user_input:
    # Split the text into the last three words (or less)
    text_input = " ".join(user_input.split()[-3:])
    
    # Get the prediction
    predicted_word = predict_word(text_input)
    
    # Display the result
    st.write(f"Predicted Next Word: **{predicted_word}**")
