from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('next_words.keras')
tokenizer = pickle.load(open('token.pkl','rb'))
def predict_word(model,tokenizer,text):
    sequence= tokenizer.texts_to_sequences([text])
    sequence= np.array(sequence)
    preds = np.argmax(model.predict(sequence),axis=-1)
    predicted_word =""
    for key,value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    print(predicted_word)
    return predicted_word
while True:
    text = input("Enter your line(type 1 to exit)")
    if text=="1":
        break
    else:
        text=text.split(" ")
        text=text[-3:]
        predict_word(model,tokenizer,text)
        print(f"Predicted next word: {predict_word}")
        
