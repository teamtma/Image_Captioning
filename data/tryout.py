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

import eval

def caption_this(image_url, max_length, afs, decoder, encoder, ifem, tokenizer):
    #### Try it on your own images ####
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension,
                                         origin=image_url)

    result, attention_plot = eval.evaluate(image_path, max_length, afs, decoder, encoder, ifem, tokenizer)
    print ('Prediction Caption:', ' '.join(result))
    eval.plot_attention(image_path, result, attention_plot)
    # opening the image
    Image.open(image_path)