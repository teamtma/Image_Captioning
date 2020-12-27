import os,sys
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import collections
import random
import re
import numpy as np
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm

import model



def split_data_training_testing(img_name_vector, cap_vector):
    #### Split the data into training and testing #####
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
      img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
      capt_len = len(img_to_cap_vector[imgt])
      img_name_train.extend([imgt] * capt_len)
      cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
      capv_len = len(img_to_cap_vector[imgv])
      img_name_val.extend([imgv] * capv_len)
      cap_val.extend(img_to_cap_vector[imgv])

    #len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)
    print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
    return img_name_train, cap_train, img_name_val, cap_val


# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap


### Creating tf dataset and the tf model
def create_tf_training_dataset(top_k, img_name_train, cap_train):
    #### Create a tf.data dataset for training ####
    # Feel free to change these parameters according to your system's configuration
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64
      
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    ### This part could potentially be separated into another function
    encoder = model.CNN_Encoder(embedding_dim)
    decoder = model.RNN_Decoder(embedding_dim, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    return dataset, encoder, decoder, optimizer, loss_object, num_steps, attention_features_shape