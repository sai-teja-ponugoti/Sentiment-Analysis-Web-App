import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import tensorflow
import re
import gc


app = Flask(__name__)

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
max_features = 20000
maxlen = 200

# function to reprocess texts by removing special charectrs and non ascii charecters
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        text = text.lower()
        # removing non alphanumeric chars
        text = NON_ALPHANUM.sub(r' ', text)
        # removing non ascii chars
        text = NON_ASCII.sub(r'', text)
        normalized_texts.append(text)
    return normalized_texts


def tokenize(test_texts):
    f = open('tokenizer_tensorflow.txt', "r")
    tokenizer_string = f.read()
    f.close()
    return tensorflow.keras.preprocessing.text.tokenizer_from_json(tokenizer_string).texts_to_sequences(test_texts)

# function to pad tokenized sequences to constant length
def padding_seq(data):
    return tensorflow.keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen, padding='post', truncating='post')


@app.route('/')
def home():
    return render_template('home.html')


def get_result(test_texts):
    test_texts = normalize_texts(test_texts)
    test_texts = tokenize(test_texts)
    test_texts = padding_seq(test_texts)
    model = tensorflow.keras.models.load_model('glove_200_model.h5')

    return round(model.predict(test_texts)[0][0] * 100, 2)

@app.route('/predict', methods = ['POST'])
def predict():
    test_texts = [x for x in request.form.values()]

    result = get_result(test_texts)

    # return render_template('home.html', prediction_text= "The sentiment is {}".format(result))
    if result>50:
        return render_template('home.html', prediction_text="The sentiment of the text is positive. Amount of positivity {}".format(result))
    else:
        return render_template('home.html', prediction_text="The sentiment of the text is negative. Amount of positivity {}".format(result))


if __name__ == '__main__':
    app.run(debug=True)