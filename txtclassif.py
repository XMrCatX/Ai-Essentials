from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataframe = pd.read_csv('amazon_pc_user_reviews.csv')
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
