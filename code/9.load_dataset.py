# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:30:01 2021

@author: jiho Ahn
@topic: load dataset with tensorflow(1)
"""
# tf.data.Dataset.from_tensor_slices
import tensorflow as tf
import numpy as np

train_x = np.arange(500).astype(np.float32).reshape(-1, 1)
train_y = 3*train_x + 1

# np.ndarray or list -> tensor dataset object
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(100).batch(32)

for x, y in train_ds:
    print(x.shape, y.shape, '\n')

# %%
# tensorflow.keras.datasets
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
'''
print(type(train_images))  # numpy.ndarray
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
'''

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(100).batch(9)

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.batch(32)


train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10), squeeze=False)

for ax_idx, ax in enumerate(axes.flat):
    image = images[ax_idx, ...]
    label = labels[ax_idx]
    
    ax.imshow(image.numpy(), 'gray')
    ax.set_title(label.numpy(), fontsize=20)

    

