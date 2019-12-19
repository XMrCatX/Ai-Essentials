from __future__ import absolute_import, division, print_function, unicode_literals
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

NUM_WORDS = 8000
SEQ_LEN = 25

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataframe = pd.read_csv('amazon_pc_user_reviews_10000.csv')
print(dataframe.head())

reviews = dataframe['review_body']
labels = dataframe['star_rating']

review_train, review_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.25, random_state=1000)

print(len(review_train), 'train examples')
print(len(review_test), 'test examples')

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(review_train.apply(lambda x: np.str_(x)))

x_train =  vectorizer.transform(review_train.apply(lambda x: np.str_(x)))
x_test = vectorizer.transform(review_test.apply(lambda x: np.str_(x)))

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)
print("Accuracy:", score)

input_dim = x_train.shape[1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(NUM_WORDS,16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max')
#callbacks=[es]
model.fit(x_train, y_train, batch_size=10, validation_data=(x_test, y_test) ,epochs=100, verbose=False)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#result = model.evaluate(test_seqs, test['star_rating'].values)

#print(result)

model.save('model.h5')
