from __future__ import absolute_import, division, print_function, unicode_literals
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
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

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataframe = pd.read_csv('amazon_pc_user_reviews.csv')
print(dataframe.head())

reviews = dataframe['review_body']
labels = dataframe['star_rating']

review_train, review_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.5)

print(len(review_train), 'train examples')
print(len(review_test), 'test examples')

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(review_train)

x_train =  tokenizer.texts_to_sequences(review_train.apply(lambda x: np.str_(x)))
x_test = tokenizer.texts_to_sequences(review_test.apply(lambda x: np.str_(x)))

vocab_size = len(tokenizer.word_index) + 1
print("vocab size: ",vocab_size)
print("Normal review:")
print(review_train[2])
print("tokenized review:")
print(x_train[2])

maxlen = 256

x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
print("na Padding:")
print(x_train[2])

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)
print("Accuracy:", score)

embedding_dim = 50

model = keras.Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=10, validation_data=(x_test, y_test) ,epochs=20, verbose=1)

loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save('model.h5')

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
