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


def download_and_prep_images():
    #### Download and prepare the MS-COCO dataset ####
    # Download caption annotation files
    annotation_folder = '/annotations/'
    if not os.path.exists(os.path.abspath('.') + '/../../data' + annotation_folder):
      annotation_zip = tf.keras.utils.get_file('captions.zip',
                                              cache_subdir=os.path.abspath('.') + '/../../data',
                                              origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                              extract = True)
      annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'
      os.remove(annotation_zip)
    else:
      annotation_file = os.path.abspath('.') + '/../../data'+'/annotations/captions_train2017.json'

    # Download image files
    image_folder = '/train2017/'
    if not os.path.exists(os.path.abspath('.') + '/../../data' + image_folder):
      image_zip = tf.keras.utils.get_file('train2017.zip',
                                          cache_subdir=os.path.abspath('.') + '/../../data',
                                          origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                          extract = True)
      PATH = os.path.dirname(image_zip) + image_folder
      os.remove(image_zip)
    else:
      PATH = os.path.abspath('.') + '/../../data' + image_folder
      return annotation_file, PATH



def limit_database_size(annotation_file, PATH):
    #### Optional: limit the size of the training set ####  
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
        
    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
      caption = f"<start> {val['caption']} <end>"
    #  image_path = PATH + 'COCO_train2017_' + '%012d.jpg' % (val['image_id'])
      image_path = PATH + '%012d.jpg' % (val['image_id'])
      image_path_to_caption[image_path].append(caption)   
        
    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will 
    # lead to 30,000 examples.
    first_N_images=100
    train_image_paths = image_paths[:first_N_images]
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
      caption_list = image_path_to_caption[image_path]
      train_captions.extend(caption_list)
      img_name_vector.extend([image_path] * len(caption_list))

    #print(img_name_vector)
    print(train_captions[0])
    im = Image.open(img_name_vector[0])
    #im.show()
    return img_name_vector, train_captions
          