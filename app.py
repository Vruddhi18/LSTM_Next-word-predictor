import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import kagglehub
import os

# Path to dataset files
path = kagglehub.dataset_download("muhammadbilalhaneef/sherlock-holmes-next-word-prediction-corpus")
print(os.listdir(path))
dataset = os.path.join(path,'Sherlock Holmes.txt')

# Load and preprocess the dataset
with open(dataset, 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # Add 1 for padding index

# Prepare the input sequences and output labels
input_sequences = []
for i in range(1, len(text.split())):
    n_gram_sequence = text.split()[i-1:i+1]
    input_sequences.append([tokenizer.word_index[word] for word in n_gram_sequence])

# Pad sequences to ensure consistent input size
max_sequence_length = 4
X = np.array([seq[:-1] for seq in input_sequences])
y = to_categorical([seq[-1] for seq in input_sequences], num_classes=total_words)

# Define and compile the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up the checkpoint callback to save the best model
checkpoint = ModelCheckpoint("next_words.keras", save_best_only=True, save_weights_only=False, monitor='loss', mode='min', verbose=1)

# Train the model
model.fit(X, y, epochs=2, batch_size=64, callbacks=[checkpoint])

# Save the trained tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Set up Streamlit page
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar setup
st.sidebar.title("Next Word Prediction Game 🎮")
st.sidebar.markdown("**Challenge yourself and see if you can guess the next word!**")

# Main app setup
st.title("Next Word Predictor 🧠")
st.markdown("### Type a sentence, and let's see if you can guess the AI's next word prediction!")

# Define the function for predicting the next word
def predict_next_word(input_text, context_length=4):
    # Preprocess the input text
    text_input = " ".join(input_text.split()[-context_length:])
    sequence = tokenizer.texts_to_sequences([text_input])
    sequence = np.array(sequence)
    
    # Predict the next word
    prediction = model.predict(sequence)
    predicted_index = np.argmax(prediction, axis=-1)
    
    # Get the predicted word
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            predicted_word = word
            break
    return predicted_word

# Game logic
if "game_started" not in st.session_state:
    st.session_state["game_started"] = False
    st.session_state["score"] = 0
    st.session_state["round"] = 1

if not st.session_state["game_started"]:
    st.markdown("Click the button below to start the game!")
    if st.button("Start Game 🎯"):
        st.session_state["game_started"] = True
        st.session_state["score"] = 0
        st.session_state["round"] = 1

if st.session_state["game_started"]:
    # Show round and score in the sidebar
    st.sidebar.write(f"**Round:** {st.session_state['round']} | **Score:** {st.session_state['score']}")

    # Input field for user to type a sentence
    user_input = st.text_input("Type a sentence:", placeholder="Type your sentence here...")

    if user_input:
        # Predict the next word
        predicted_word = predict_next_word(user_input)
        st.session_state["predicted_word"] = predicted_word
        st.markdown(f"### AI's predicted word: **{predicted_word}**")

    # Allow user to guess the predicted word
    if "predicted_word" in st.session_state:
        guess = st.text_input("What's your guess?", placeholder="Type your guess here...")

        if guess:
            if guess.lower() == st.session_state["predicted_word"]:
                st.success(f"🎉 Correct! The predicted word is **{st.session_state['predicted_word']}**.")
                st.session_state["score"] += 1
            else:
                st.error(f"❌ Oops! The correct word was **{st.session_state['predicted_word']}**.")
            st.session_state["round"] += 1
            st.session_state.pop("predicted_word", None)  # Reset predicted word after each round

    # Option to restart the game
    st.sidebar.markdown("### 🎲 Game Controls")
    if st.sidebar.button("Restart Game"):
        st.session_state["game_started"] = False
        st.session_state["score"] = 0
        st.session_state["round"] = 1

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Built with 💻 by AI Enthusiasts")
