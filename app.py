from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Charger les données
data = pd.read_parquet("data.parquet")

# Prétraitement des données
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["text"])
total_words = len(tokenizer.word_index) + 1

# Chargement du modèle
model = load_model('bamanankan_next_word_prediction_model.h5')

@app.route('/predict', methods=['POST'])
def predict_next_word():
    text = request.json['text']
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    return jsonify({"next_word": output_word})


if __name__ == '__main__':
    app.run(debug=True)
