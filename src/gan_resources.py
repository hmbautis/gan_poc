import os
from glob import glob
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import wand.image as wi
from PIL import Image

# These functions serve to process the image data into tensors.
def load_image(image_path):
    max_dim=512
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img

def handle_format(image_path):
    img=wi.Image(filename=image_path)
    img.format='jpg'
    new_path = str(image_path).replace('.heic','.jpg')
    img.save(filename=new_path)
    img.close()
    return new_path

# This function serve to display the image
def imshow(image, title=None):
    if(len(image.shape) > 3):
        image=np.squeeze(image, axis=0)
    plt.imshow(image)
    if(title):
        plt.title(title)

def my_model(layer_names):
    vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    outputs = [vgg_model.get_layer(name).output for name in layer_names]
    model=tf.keras.Model([vgg_model.input], outputs)
    return model

def gram_matrix(input_tensor): 
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) 
    return result/(num_locations)

def total_cost(outputs,style_weight,style_weights,style_targets,style_layers,content_weight,content_layers,content_targets):
    style_outputs=outputs['style']
    content_outputs=outputs['content']
    style_loss=tf.add_n([style_weights[name]*tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                        for name in style_outputs.keys()])
    style_loss*=style_weight/len(style_layers)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss*=content_weight/len(content_layers)
    loss=style_loss+content_loss
    return loss

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)