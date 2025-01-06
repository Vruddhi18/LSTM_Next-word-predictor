from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import pickle
import gdown
import os

# Initialize Flask app
app = Flask(__name__)

# Paths for model and tokenizer
output_model_tflite = "next_words.tflite"
output_tokenizer = "token.pkl"

# Download model and tokenizer files from Google Drive if not already present
if not os.path.exists(output_model_tflite):
    url_model = "https://drive.google.com/uc?id=1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG"
    output_model = "next_words.keras"
    gdown.download(url_model, output_model, quiet=False)

    # Convert model to TensorFlow Lite format
    from tensorflow import lite
    keras_model = load_model(output_model)
    converter = lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [lite.Optimize.DEFAULT]  # Enable optimizations
    tflite_model = converter.convert()

    # Save the optimized model
    with open(output_model_tflite, 'wb') as f:
        f.write(tflite_model)

if not os.path.exists(output_tokenizer):
    gdown.download(
        "https://drive.google.com/uc?id=1VmxkGEFA6XQV3guHviwgHlb55EpHa6aG", 
        output_tokenizer, 
        quiet=False
    )

# Load the optimized TensorFlow Lite model
interpreter = Interpreter(model_path=output_model_tflite)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load tokenizer
with open(output_tokenizer, 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict the next word
def predict_word_lite(interpreter, tokenizer, text):
    # Preprocess input
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence, dtype=np.float32)  # Ensure float32 type

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sequence)

    # Perform inference
    interpreter.invoke()

    # Get prediction results
    preds = np.argmax(interpreter.get_tensor(output_details[0]['index']), axis=-1)
    
    # Map prediction to word
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
        predicted_word = predict_word_lite(interpreter, tokenizer, text)
        return jsonify({"predicted_word": predicted_word})
    else:
        return jsonify({"error": "No text provided"}), 400

if __name__ == "__main__":
    app.run(debug=True)
