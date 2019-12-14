from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

PCrev_train, info = tfds.load(name="amazon_us_reviews/PC_v1_00", with_info=True , split="train")
assert isinstance(PCrev_train, tf.data.Dataset)
print(info)

for test_example in PCrev_train:  # Only take a single example
  item, categorie, review, stars = test_example["data"]["product_title"], test_example["data"]["product_category"], test_example["data"]["review_body"], test_example["data"]["star_rating"]
  print("naam: ", item.numpy(),"cat: ", categorie.numpy(),"review: ",review.numpy(),"stars: ",stars.numpy())
