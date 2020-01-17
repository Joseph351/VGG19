# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:28:36 2019

@author: Joseph Duodu
"""
from __future__ import print_function
import tensorflow as tf

input_name = "vgg19_input"
def data_input_pipeline(filenames, apply_batch=False, perform_shuffle=False, repeat_count=None, batch_size=None):
    def _parse_function(serialized):
        features = \
        {
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64)
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        width = tf.cast(parsed_example['image/width'], tf.int64)
        height = tf.cast(parsed_example['image/height'], tf.int64)
        #channels = tf.cast(parsed_example['image/channels'], tf.int64)
        image_shape = tf.stack([height, width, 3])
        image_raw = parsed_example['image/encoded']
        label = tf.cast(parsed_example['image/class/label'], tf.float32)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.io.decode_jpeg(image_raw, 3)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, image_shape)
        
        #resize and crop so that images are 224 by 224
        resized_image = tf.image.resize_with_crop_or_pad(image, 224, 224)
        
        d = dict(zip([input_name], [resized_image])), [label]
        #d = [resized_image], [label]
        return d
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=1251)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    if apply_batch:
        dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    #batch_features, batch_labels = iterator.get_next()
    return  iterator    #batch_features["vgg19_input"], batch_labels