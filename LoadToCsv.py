# What version of Python do you have?
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import csv
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Loading Dataset: amazon_us_reviews/PC_v1_00 from Tensorflow")
Data, info = tfds.load(name="amazon_us_reviews/PC_v1_00", with_info=True, split="train")
assert isinstance(Data, tf.data.Dataset)
print("Dataset info: ", info)
csv_file = "amazon_pc_user_reviews.csv"
try:
    with open(csv_file, 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["product_title","review_body","star_rating"])
        for data in Data.take(1000000):
            item, review, stars = data["data"]["product_title"], data["data"]["review_body"], data["data"]["star_rating"]
            writer.writerow([item.numpy().decode("utf-8"),review.numpy().decode("utf-8"),stars.numpy()])
        print("Dataset saved with name: amazon_pc_user_reviews.csv")
except IOError:
    print("I/O error")
