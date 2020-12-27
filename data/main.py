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

### File Imports
import image_prep
import image_process
import cap_token
import data_prep
import model
import train
import eval
import tryout


annotation_file, PATH = image_prep.download_and_prep_images()
img_name_vector, train_captions = image_prep.limit_database_size(annotation_file, PATH)

ifem=image_process.proces_with_InceptionV3(img_name_vector)

# Choose the top top_k words from the vocabulary
top_k = 5000
cap_vector, tokenizer, max_length = cap_token.tokenize_captions(top_k, train_captions)


img_name_train, cap_train, img_name_val, cap_val = data_prep.split_data_training_testing(img_name_vector, cap_vector)
dataset, encoder, decoder, optimizer, loss_object, num_steps, afs = data_prep.create_tf_training_dataset(top_k, img_name_train, cap_train)

start_epoch = 0
start_epoch, ckpt_manager = model.checkpoint(start_epoch, encoder, decoder, optimizer)

#### Training ####
EPOCHS = 20
loss_plot = train.epoch_loop(start_epoch, EPOCHS, dataset, decoder, encoder, optimizer, tokenizer, loss_object, num_steps, ckpt_manager)
train.plot_loss(loss_plot)

### Evaluate the validation dataset ####
eval.validation_set_captions(img_name_val, cap_val, tokenizer, max_length, afs, decoder, encoder,ifem)


### Try out your own images
image_url = 'https://tensorflow.org/images/surf.jpg'
#image_url = 'https://www.gstatic.com/webp/gallery3/1.png'
tryout.caption_this(image_url, max_length, afs, decoder, encoder, ifem, tokenizer)
print('endofmain')


