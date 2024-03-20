from flask import Flask, render_template, request,jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load the Keras model
model = load_model('chat_model1.keras')

# Maximum sequence length for padding
max_len = 20

with open("intents.json") as file:
    data = json.load(file)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for chatbot response
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = request.json['user_input']

    # Load the tokenizer from the pickled file
    with open('tokenizer1.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoder from the pickled file
    with open('label_encoder1.pickle', 'rb') as handle:
        lbl_encoder = pickle.load(handle)

#    # Convert user input to sequence of integers
    sequence = tokenizer.texts_to_sequences([user_input])

    # Pad sequences to a fixed length
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Make prediction using the Keras model
    result = model.predict(padded_sequence)

    # Get the predicted label
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in range(len(data['intents'])):
        t = data['intents'][i]['tag']
        responses_options = data['intents'][i]['responses']
        if(tag==t):
            responses = np.random.choice(responses_options)
    return jsonify({'response': responses})
    
if __name__ == '__main__':
    app.run(debug=True)

