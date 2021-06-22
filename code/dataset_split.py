# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 22:05:19 2021

@author: jiho Ahn
@topic: Dataset Split
"""
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
import numpy as np 

train_x = np.arange(100).reshape(-1, 1)
train_y = 3*train_x + 1

train_validation_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

#tmp_ds = train_ds.take(10)  # 10개만 추출해서 새로운 dataset을 만듦
#
#for x, y in tmp_ds:
#    print(x)
#    print(y, '\n')

n_train_validation = 100
train_ratio = 0.8
n_train = int(n_train_validation * train_ratio)

train_ds = train_validation_ds.take(n_train)

# %%

n_train_validation = 100
train_ratio = 0.8
n_train = int(n_train_validation * train_ratio)
n_validation = n_train_validation - n_train

train_x = np.arange(100).reshape(-1, 1)
train_y = 3*train_x + 1

train_validation_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

remaining_ds = train_validation_ds.skip(n_train)  # 앞에서 n_train개를 건너뛰고 dataset을 생성
validation_ds = remaining_ds.take(n_validation)

for x, y in validation_ds:
    print(x)
    print(y, '\n')

# %%
(train_validation_ds, test_ds), ds_info = tfds.load(name='mnist',
                                                    shuffle_files=True,
                                                    as_supervised=True,
                                                    split=['train', 'test'],
                                                    with_info=True)

n_train_validation = ds_info.splits['train'].num_examples
train_ratio = 0.8
n_train = int(n_train_validation * train_ratio)
n_validation = n_train_validation - n_train

train_ds = train_validation_ds.take(n_train)
remaining_ds = train_validation_ds.skip(n_train)
validation_ds = remaining_ds.take(n_validation)

train_ds = train_ds.shuffle(100).batch(32)
validation = train_ds.batch(32) 
test_ds = test_ds.batch(32)