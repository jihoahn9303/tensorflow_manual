# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 22:27:59 2021

@author: jiho Ahn
@topic: Dataset Preprocessing
"""

import tensorflow as tf
import tensorflow_datasets as tfds

train_ds = tfds.load(name='mnist',
                     shuffle_files=True,
                     as_supervised=True,
                     split='train',
                     batch_size=4)

for images, labels in train_ds:
    print(images.shape)
    print(images.dtype)  # uint8
    
    print(labels.shape)
    print(labels.dtype)  # int64
    break

# %%
# 1. 원활한 학습을 위해, images data type을 float로 변경해야 한다.
# 2. 마찬가지 이유로, images data의 정규화 과정이 필요하다. (weight의 변동성을 최소화)

def standardization(images, labels):
    images = tf.cast(images, tf.float32) / 255.
    return [images, labels]

train_ds = tfds.load(name='mnist',
                     shuffle_files=True,
                     as_supervised=True,
                     split='train',
                     batch_size=4)

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print(images.dtype, tf.reduce_max(images))

train_ds = train_ds.map(standardization)  # go standardization
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print(images.dtype, tf.reduce_max(images))

     