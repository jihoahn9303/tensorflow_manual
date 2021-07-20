# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 23:03:12 2021

@author: jiho Ahn
@topic: Losses and Metrics
"""
import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# %%
# Binary Cross Entropy(1) : scalar
loss_object = BinaryCrossentropy()

predictions = np.array([0.3]).reshape(-1, 1)
labels = np.array([1])

loss = loss_object(labels, predictions)
loss_manual = -1*(labels*np.log(predictions) + \
                  (1 - labels)*np.log(1 - predictions))

print(loss.numpy())
print(loss_manual)

# %%
# Binary Cross Entropy(2) : one-hot vector
predictions = np.array([0.3, 0.7]).reshape(-1, 1)
labels = np.array([1, 0]).reshape(-1, 1)

loss = loss_object(labels, predictions)
loss_manual = -1*(labels*np.log(predictions) + \
                  (1 - labels)*np.log(1 - predictions))
loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)

# %%
# Binary Cross Entropy(3) : a bunch of one-hot vectors(one-hot matrix)
predictions = np.array([[0.3, 0.7], [0.4, 0.6], [0.1, 0.9]])
labels = np.array([[0, 1], [1, 0], [1, 0]])

loss = loss_object(labels, predictions)
'''
loss_manual = -1*(labels*np.log(predictions) + \
                  (1 - labels)*np.log(1 - predictions))
loss_manual = np.mean(loss_manual, axis=1)
loss_manual = np.mean(loss_manual)
'''
# this calculation outputs same results
loss_manual = -1*labels*np.log(predictions)
loss_manual = np.sum(loss_manual, axis=1).mean()

'''
for example) predictions = np.array[0.3, 0.7], labels = np.array[0, 1]
-> 정답은 1이라는 뜻이다.
-> 따라서 log cross entropy 공식에 따라, loss = -(1*log0.7 + 0*log0.3) 이다.
-> 이를 numpy 관점에서 표현하면, -1*[0, 1]*log[0.3, 0.7] = [-0, -log0.7] 이다.
-> 따라서 loss는 -1*np.sum(labels*np.log(predictions), axis=1)로 간단하게 표현 가능하다.
'''

print(loss.numpy())
print(loss_manual)

# %%
# Categorical cross entropy
loss_object = CategoricalCrossentropy()

predictions = np.array([[0.2, 0.1, 0.7],
                       [0.4, 0.3, 0.3],
                       [0.1, 0.8, 0.1]])

labels = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

loss = loss_object(labels, predictions)
loss_manual = -1*labels*np.log(predictions)
loss_manual = np.sum(loss_manual, axis=1).mean()

print(loss.numpy())
print(loss_manual)

# %%
# Sparse Categorical cross entropy
'''
이전 셀에서의 문제점 
labels = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])인 경우,
그냥 정답을 2, 1, 0으로 하면 되지, 왜 굳이 원핫 인코딩으로 정답을 표현했는가? 
'''

loss_object = SparseCategoricalCrossentropy()
predictions = np.array([[0.2, 0.1, 0.7],
                       [0.4, 0.3, 0.3],
                       [0.1, 0.8, 0.1]])

# equals np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
labels = np.array([2, 1, 0]) 

# SparseCategoricalCrossentropy()의 경우, 인자를 tensor 형태로 받는다.
loss = loss_object(tf.constant(labels), tf.constant(predictions))

# loss_manual calculation process
loss_manual = np.zeros(shape=(labels.shape[0],))
for idx in range(predictions.shape[0]):
    loss_manual[idx] = -np.log(predictions[idx][labels[idx]])
    '''
    ex) 반복문 첫 루프의 경우, 정답은 3번째 클래스 이므로, loss 식은 아래와 같다.
    loss = -(0*log0.2 + 0*log0.1 + 1*log0.7) = -1*log0.7
    이 계산된 loss 값 loss_manual vector의 첫 번째 원소로 담아주는 것이다.
    '''
loss_manual = np.mean(loss_manual)

print(loss.numpy())
print(loss_manual)

# %%

import tensorflow_datasets as tfds

train_ds = tfds.load(name='mnist',
                     as_supervised=True,
                     shuffle_files=True,
                     split='train')

train_ds = train_ds.batch(8)

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

# 이 경우, loss function은  SparseCategoricalCrossentropy로 사용해야 함
print(labels)  
# %%
# Categorical Accuracy
metric = CategoricalAccuracy()
predictions = np.array([[0.2, 0.2, 0.6],
                       [0.1, 0.8, 0.1]])

labels = np.array([[0, 0, 1], [0, 1, 0]]) 

acc = metric(labels, predictions)
print(acc*100)

# %%
# Sparse Categorical Accuracy
metric = SparseCategoricalAccuracy()
predictions = np.array([[0.2, 0.2, 0.6],
                       [0.1, 0.8, 0.1]])

labels = np.array([0, 2]) 
acc = metric(labels, predictions)
print(acc*100)