# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:03:18 2021

@author: jiho Ahn
"""

import tensorflow as tf
import numpy as np

t1 = tf.constant([1, 2, 3])
t2 = tf.constant([10, 20, 30])

print(t1 + t2)

# %%

t1 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)
t2 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)

t1 = tf.cast(t1, dtype=tf.int16)
t2 = tf.cast(t2, dtype=tf.int16)

print(t1.numpy())
print(t2.numpy())

print(t1 + t2)

# %%
# broadcasting 1

t1 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)
t2 = tf.random.normal(shape=(1, 4), mean=0, stddev=5)

t1 = tf.cast(t1, dtype=tf.int16)
t2 = tf.cast(t2, dtype=tf.int16)

t3 = t1 + t2

print(t1.numpy(), '\n')
print(t2.numpy(), '\n')
print(t3.numpy(), '\n')

# %%
# broadcasting 2

t1 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)
t2 = tf.random.normal(shape=(3, 1), mean=0, stddev=5)

t1 = tf.cast(t1, dtype=tf.int16)
t2 = tf.cast(t2, dtype=tf.int16)

t3 = t1 + t2

print(t1.numpy(), '\n')
print(t2.numpy(), '\n')
print(t3.numpy(), '\n')

# %%
# broadcasting 3
t1 = tf.random.normal(shape=(3, 1), mean=0, stddev=5)
t2 = tf.random.normal(shape=(1, 4), mean=0, stddev=5)

t1 = tf.cast(t1, dtype=tf.int16)
t2 = tf.cast(t2, dtype=tf.int16)

t3 = t1 + t2

print(t1.numpy(), '\n')
print(t2.numpy(), '\n')
print(t3.numpy(), '\n')

# %% 

tf.reduce_sum
tf.reduce_prod
tf.reduce_max
tf.reduce_min
tf.reduce_mean
tf.reduce_std
tf.reduce_variance
tf.reduce_all
tf.reduce_any

# %%
t1 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)
t1 = tf.cast(t1, dtype=tf.int16)

t2 = tf.reduce_sum(t1)
print(t1.numpy())
print(t2.numpy())

# %%
t2 = tf.reduce_sum(t1, axis=0)  # 첫 번째 축을 없애면서 더한다.
print(t1.numpy())
print(t2.numpy())

# %%
t1 = tf.random.normal(shape=(128, 128, 3), mean=0, stddev=5)
t1 = tf.cast(t1, dtype=tf.int16)

t2 = tf.reduce_sum(t1, axis=2)
print(t1.shape, '\n')
print(t2.shape)



