# from tensorflow.keras.models import load_model
# import numpy as np
# import pickle

# model = load_model('next_words.keras')
# tokenizer = pickle.load(open('token.pkl','rb'))
# def predict_word(model,tokenizer,text):
#     sequence= tokenizer.texts_to_sequences([text])
#     sequence= np.array(sequence)
#     preds = np.argmax(model.predict(sequence),axis=-1)
#     predicted_word =""
#     for key,value in tokenizer.word_index.items():
#         if value == preds:
#             predicted_word = key
#             break
#     print(predicted_word)
#     return predicted_word
# while True:
#     text = input("Enter your line(type 1 to exit)")
#     if text=="1":
#         break
#     else:
#         text=text.split(" ")
#         text=text[-3:]
#         predict_word(model,tokenizer,text)
#         print(f"Predicted next word: {predict_word}")
#####################
# pip install transformers datasets torch
from datasets import load_dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments

# Load the WikiText-2 dataset (smaller dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Select a small subset of the dataset
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(200))  
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(50))    

# Load GPT-Neo tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

# Set the pad token to be the same as the eos token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True)
    # Shift the input_ids by 1 to create the labels
    inputs['labels'] = inputs['input_ids'].clone()
    return inputs
tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_test = small_test_dataset.map(tokenize_function, batched=True)

# Load GPT-Neo model (125M version)
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    fp16=True,  # Using mixed precision for faster training.
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Train the model
trainer.train()

# Prediction function
def predict_next_word(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=len(inputs[0]) + 1, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split()[-1]

# Test the prediction function
print(predict_next_word("The quick brown"))
