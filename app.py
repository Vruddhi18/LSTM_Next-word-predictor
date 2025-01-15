import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import random
import os
import tensorflow as tf
import gdown

# Configure the app
st.set_page_config(
    page_title="Guess the Next Word!",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Google Drive file ID
file_id = '1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG'  
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Download model to the current working directory
output_path = 'next_words.keras'
gdown.download(url, output_path, quiet=False)

# Load the Keras model

# Check if the model is loaded
print("Model loaded successfully!")

try:
    model = load_model(output_path)
    tokenizer = pickle.load(open('token.pkl', 'rb'))
except Exception as e:
    st.error("Error loading model or tokenizer. Please check your files.")
    st.stop()

# Prediction function
def predict_word(text, context_length):
    text_input = " ".join(text.split()[-context_length:])
    sequence = tokenizer.texts_to_sequences([text_input])
    sequence = np.array(sequence)
    preds = model.predict(sequence)
    predicted_index = np.argmax(preds, axis=-1)
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break
    return predicted_word

# Sidebar for navigation
st.sidebar.title("🎮 Guess the Next Word!")
st.sidebar.markdown("**Challenge yourself and guess the AI's next word prediction!**")

# App title
st.title("🎮 Guess the Next Word!")
st.markdown("### Enter a sentence, and let's see if you can guess what the AI predicts next!")

# Game setup
if "game_started" not in st.session_state:
    st.session_state["game_started"] = False
    st.session_state["score"] = 0
    st.session_state["round"] = 1

# Start the game
if not st.session_state["game_started"]:
    st.markdown("Click the button below to start the game!")
    if st.button("Start Game 🎯"):
        st.session_state["game_started"] = True
        st.session_state["score"] = 0
        st.session_state["round"] = 1
# Game logic
if st.session_state["game_started"]:
    st.sidebar.write(f"**Round:** {st.session_state['round']} | **Score:** {st.session_state['score']}")

    # User input
    user_input = st.text_input("Type a sentence:", placeholder="Type your sentence here...")

    
    # Predict the word
    if user_input and st.button("Predict 🔮"):
        with st.spinner("AI is predicting..."):
            predicted_word = predict_word(user_input,4)
            st.session_state["predicted_word"] = predicted_word
            st.markdown("### 📝 Make your guess!")

    # Guess the word
    if "predicted_word" in st.session_state:
        guess = st.text_input("What's your guess?")
        if guess:
            if guess.lower() == st.session_state["predicted_word"]:
                st.success(f"🎉 Correct! The predicted word is **{st.session_state['predicted_word']}**.")
                st.session_state["score"] += 1
            else:
                st.error(f"❌ Oops! The correct word was **{st.session_state['predicted_word']}**.")
            st.session_state["round"] += 1
            st.button("Next Round 🔄", on_click=lambda: st.session_state.pop("predicted_word", None))

    # Scoreboard and restart option
    st.sidebar.markdown("### 🎲 Game Controls")
    if st.sidebar.button("Restart Game"):
        st.session_state["game_started"] = False
        st.session_state["score"] = 0
        st.session_state["round"] = 1

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Built with 💻 by AI Enthusiasts")
