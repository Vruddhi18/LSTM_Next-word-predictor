from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('next_words.keras')
tokenizer = pickle.load(open('token.pkl', 'rb'))

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    text = text.split(" ")
    text = text[-3:]
    predicted_word = predict_word(model, tokenizer, text)
    return jsonify({'predicted_word': predicted_word})

@app.route('/')
def home():
    return "Next-Word Predictor is running!"

if __name__ == "__main__":
    app.run(debug=True)
