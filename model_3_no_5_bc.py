# Deze file gebruikt model 3 met de Dataset zonder 5 ster reviews, binary_crossentropy en activation layer softmax

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.sequence import pad_sequences

NUM_WORDS = 8000
SEQ_LEN = 25
maxlen = 256

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Laad de dataframe met het csv.bestand
dataframe = pd.read_csv('amazon_pc_user_reviews_no_5.csv')
print(dataframe.head())

# Laad de stopwoorden lijst
stop_words = set(stopwords.words('english'))
pat = r'\b(?:{})\b'.format('|'.join(stop_words))

# Verwijderen van de stopwoorden in de dataset
print(dataframe.iloc[1])
dataframe['review_body'] = dataframe['review_body'].str.replace(pat, '')
print(dataframe.iloc[1])
dataframe['review_body'] = dataframe['review_body'].str.replace(r'\s+', ' ')
print(dataframe.iloc[1])

reviews = dataframe['review_body']
labels = dataframe['star_rating']

# Woorden index maken van de overgebeleven woorden
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(reviews.apply(lambda x: np.str_(x)))

vocab_size = len(tokenizer.word_index) + 1
print("vocab size: ",vocab_size)
print(tokenizer.word_index)

# Tokenizen van de dataset
review_token =  tokenizer.texts_to_sequences(reviews.apply(lambda x: np.str_(x)))
review_normal = pad_sequences(review_token, padding='post', maxlen=maxlen)

# Test, Train en Validatie sets maken
review_train, review_validation, y_train, y_validation = train_test_split(review_normal, labels, test_size=0.25 ,shuffle=True)
review_validation, review_test, y_validation, y_test = train_test_split(review_validation, y_validation, test_size=0.25)

print(len(review_train), 'train examples')
print(len(review_test), 'test examples')


print("na Padding:")
print(review_train[0,:])

# Accuracy prediction
classifier = LogisticRegression()
classifier.fit(review_train, y_train)
score = classifier.score(review_test, y_test)
print("Accuracy:", score)

embedding_dim = 50

# Opstellen van het model
model = keras.Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(review_train, y_train, batch_size=100, validation_data=(review_validation, y_validation) ,epochs=10, verbose=1)

# Toonen van de train en test accuracy
loss, accuracy = model.evaluate(review_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(review_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Opslaan van het model
model.save('model.h5')

# Grafiek vooor het visualiseren van de progressie
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)
