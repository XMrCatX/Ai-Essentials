from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras
from sklearn.model_selection import train_test_split

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Data, info = tfds.load(name="amazon_us_reviews/PC_v1_00", with_info=True, split="train")
assert isinstance(Data, tf.data.Dataset)

for test_example in Data.take(1):  # Only take a single example
  item, categorie, review, stars = test_example["data"]["product_title"], test_example["data"]["product_category"], test_example["data"]["review_body"], test_example["data"]["star_rating"]
  print("naam: ", item.numpy().decode("utf-8"),"cat: ", categorie.numpy().decode("utf-8"),"review: ",review.numpy().decode("utf-8"),"stars: ",stars.numpy())

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = "relu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
x_val = Data.range(10000)
x_train = Data.range(10000,20000)

y_val = Data.range(10000)
y_train = Data.range(10000,20000)

fitmodel= model.fit(x_train.shuffle(buffer_size = 10000,seed=None, reshuffle_each_iteration=None), epochs=40,validation_data=(x_val, y_val), verbose=1)

result = model.evaulate(Data)
model.save("model.h5")
print(result)
