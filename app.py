from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import gdown

# Initialize Flask app
app = Flask(__name__)

# Download model and tokenizer files from Google Drive
url_model = "https://drive.google.com/file/d/1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG/view?usp=drive_link"  
output_model = "next_words.keras"
output_tokenizer = "token.pkl"

gdown.download(url_model, output_model, quiet=False)
gdown.download("https://drive.google.com/file/d/1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG/view?usp=drive_link", output_tokenizer, quiet=False) 

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

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input from the request
    text = data.get("text", "")  # Get the text input
    if text:
        text = text.split(" ")
        text = text[-3:]  # Use the last 3 words for prediction
        predicted_word = predict_word(model, tokenizer, text)
        return jsonify({"predicted_word": predicted_word})
    else:
        return jsonify({"error": "No text provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)
