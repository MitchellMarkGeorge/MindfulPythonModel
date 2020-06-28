from google.cloud import storage
from tensorflow import keras
import os
import json
CLASS_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
storage_client = storage.Client()

bucket = storage_client.get_bucket('mindful_model_tokenizer')

model_blob = bucket.blob('model.h5')
tokenizer_blob = bucket.blob('tokenizer.json')

model_blob.download_to_filename('/tmp/model.h5')
tokenizer_blob.download_to_filename('/tmp/tokenizer.json')

model = keras.models.load_model('/tmp/model.h5')

with open('/tmp/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

def classify(request):
    if request.method == 'POST':
        # pass

        request_json = request.get_json()
        
        text_list = request_json["text"]
        encoded_result = encode_text(text_list)

        result = model.predict(encoded_result)[0]

        response = {'prediction': [ {'label': class_name, 'prediction': float(result[i]) } for i, class_name in enumerate(CLASS_LIST) ]}
    
        return json.dumps(response)
        # SHOULD NOT BE SENDING STRING
    else: return "Error: POST request needed to access this API"
    




def encode_text(text_list):
    sequence = keras.preprocessing.sequence
    
    # MAX_FEATURES = 20000
    MAX_TEXT_LENGTH = 200

    # tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    # tokenizer.fit_on_texts(text_list)
    list_tokenized = tokenizer.texts_to_sequences(text_list)
    # list_tokenized_test = tokenizer.texts_to_sequences(test_raw_text)

    # xtest = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_TEXT_LENGTH)
    x = sequence.pad_sequences(list_tokenized, maxlen=MAX_TEXT_LENGTH)

    return x