import tensorflow as tf
import pandas as pd
from tensorflow import keras

# model = keras.Sequential()

import os
import io
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# should save tokenizer

# from tensorflow.keras import text, sequence
text = keras.preprocessing.text
sequence = keras.preprocessing.sequence


MAX_FEATURES = 20000
MAX_TEXT_LENGTH = 200
CLASS_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')

train_raw_text = train_dataset["comment_text"]
test_raw_text = test_dataset["comment_text"]

tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(train_raw_text))
list_tokenized_train = tokenizer.texts_to_sequences(train_raw_text)
list_tokenized_test = tokenizer.texts_to_sequences(test_raw_text)

xtest = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_TEXT_LENGTH)
xtrain = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_TEXT_LENGTH)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))


y_train = train_dataset[CLASS_LIST].values

model = keras.Sequential()

inp = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))

# model = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))(inp)
model = keras.layers.Embedding(MAX_FEATURES, 128)(inp)
model = keras.layers.LSTM(60, return_sequences=True)(model)
model = keras.layers.GlobalMaxPool1D()(model)
model = keras.layers.Dropout(0.1)(model)
model = keras.layers.Dense(50, activation="relu")(model)
model = keras.layers.Dropout(0.1)(model)
model = keras.layers.Dense(6, activation="sigmoid")(model)

model = keras.models.Model(inputs=inp, outputs=model)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(xtrain, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

print(model.predict(xtest))

# model.evaluate

model.save('model.h5', overwrite=True)
