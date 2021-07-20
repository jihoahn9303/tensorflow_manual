# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:00:40 2021

@author: jiho Ahn
@topic: Gradient Vanishing Problem
"""

import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

import json 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from termcolor import colored 

import tensorflow as tf 
import tensorflow_datasets as tfds 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

train_ds, ds_info = tfds.load(name='mnist',
                              shuffle_files=True,
                              as_supervised=True,
                              split='train',
                              with_info=True)

def normalization(images, labels):
    images = tf.cast(images, tf.float32) / 255.
    return [images, labels]


n_layer = 7
cmap = cm.get_cmap('rainbow', lut=n_layer)
units = [10] * n_layer

model = Sequential() 
model.add(Flatten())
for layer_idx in range(n_layer-1):
    model.add(Dense(units=units[layer_idx], activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

model.build(input_shape=(None, 28, 28, 1))
#model.summary()

train_batch_size = 10
train_ds = train_ds.map(normalization).batch(train_batch_size)

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD()

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

#print(images.shape, labels.shape)

with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
    
gradients = tape.gradient(loss, model.trainable_variables)

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_yscale('log')
for grad_idx, grad in enumerate(gradients[::2]):
    if grad_idx >= 1:
        grad_abs = np.abs(grad.numpy().flat)
        ax.plot(grad_abs, label='layer {}'.format(grad_idx),
                color=cmap(grad_idx),
                alpha=0.8)
        
ax.legend(bbox_to_anchor=(1, 0.5),
          loc='center left',
          fontsize=20)

fig.tight_layout()
    
#print(type(gradients))
#print(len(gradients))
#print(type(gradients[0]))
#print(gradients[0].shape)
