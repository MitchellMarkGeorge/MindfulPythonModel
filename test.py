from tensorflow import keras
import tensorflowjs as tfjs
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def encode_text(text_list: list):

    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras.preprocessing.text.tokenizer_from_json(data)

    # text = keras.preprocessing.text
    sequence = keras.preprocessing.sequence
    
    # MAX_FEATURES = 20000
    MAX_TEXT_LENGTH = 200

    # tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    # tokenizer.fit_on_texts(text_list)
    list_tokenized_train = tokenizer.texts_to_sequences(text_list)
    # list_tokenized_test = tokenizer.texts_to_sequences(test_raw_text)

    # xtest = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_TEXT_LENGTH)
    xtrain = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_TEXT_LENGTH)
    return xtrain


model = keras.models.load_model('model.h5')

# tfjs.converters.save_keras_model(model, 'tfjs-model')
# print(model.predict(encode_text(['You are the next school shooter'])).astype(float))
# model.predict(encode_text(['You are the next school shooter']))
# json.dumps(model.predict(encode_text(['You are the next school shooter'])).item())

