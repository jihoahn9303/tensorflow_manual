# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:50:41 2021

@author: jiho Ahn
"""

import tensorflow as tf
import numpy as np

#test_list = [1, 1, 1, 1, 1, 1]

#t1 = tf.constant(test_list)

t2 = tf.ones(shape=(100, 3))
#print(t2)

t2 = tf.ones(shape=(128, 128, 3))

t3 = tf.zeros(shape=(128, 128, 3))

# %%
PI = np.pi
t4 = PI * tf.ones(shape=(128, 128, 3))
#print(t4)

# %%

test_list = [[1, 2, 3], [4, 5, 6]]

t1 = tf.Variable(test_list)
#print(t1)

t2 = tf.ones_like(t1)
#print(t2)

t3 = tf.zeros_like(t1)
#print(t3)

# %%
#np.random.seed(0)
tf.random.set_seed(0)

t1 = tf.random.normal(shape=(3, 3))
#print(t1)

# %%
import matplotlib.pyplot as plt

t2 = tf.random.normal(mean=3, stddev=1, shape=(1000,))
print(t2)

fig, ax = plt.subplots(figsize=(15, 15))
ax.hist(t2.numpy(), bins=30)

ax.tick_params(labelsize=20)

# %%
t2 = tf.random.uniform(shape=(1000,), minval=-10, maxval=10)

fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(t2.numpy(), bins=30)

ax.tick_params(labelsize=20)

# %%
t2 = tf.random.poisson(shape=(1000,), lam=5)

fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(t2.numpy(), bins=30)

ax.tick_params(labelsize=20)

# %%

t1 = tf.random.normal(shape=(128, 128, 3))

print("t1.shape: ", t1.shape)
print("t1.dtype: ", t1.dtype)

# %%

test_np = np.random.randint(-10, 10, size=(100,))
print(test_np.dtype)

t1 = tf.constant(test_np)
print(t1.dtype)

# %%

test_np = np.random.randint(-10, 10, size=(100,))
print(test_np.dtype)

t1 = tf.constant(test_np, dtype=tf.float16)
print(t1.dtype)



















