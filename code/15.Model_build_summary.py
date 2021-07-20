# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:51:24 2021

@author: jiho Ahn
@topic: Model build and Summary
"""

import os 
import json
import numpy as np 

import tensorflow as rf 
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

# manage gpu resource
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %%

model = Sequential()
model.add(Flatten())
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

# summary() method를 사용하기 위해서, build() method를 통해,
# input data의 shape을 지정해줘야 한다.
model.build(input_shape=(None, 28, 28, 1))

model.summary()
# tf.keras.backend.clear_session()

# %%

class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.flatten = Flatten()
        self.d1 = Dense(units=10)
        self.d1_act = Activation('relu')
        self.d2 = Dense(units=2)
        self.d2_act = Activation('softmax')
        

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d1_act(x)
        x = self.d2(x)
        x = self.d2_act(x)
        return x
    

model = TestModel()
if not model.built:
    model.build(input_shape=(None, 28, 28, 1))

model.summary()

# %%

model = Sequential()
model.add(Flatten())
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

print(model.built)
test_img = tf.random.normal(shape=(1, 28, 28, 1))
model(test_img)
print(model.built)

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense, Activation

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=3, padding='valid', name='conv_1'))
model.add(MaxPooling2D(pool_size=2, strides=2, name='conv_1_maxpool'))
model.add(Activation('relu', name='conv_1_act'))

model.add(Conv2D(filters=10, kernel_size=3, padding='valid', name='conv_2'))
model.add(MaxPooling2D(pool_size=2, strides=2, name='conv_2_maxpool'))
model.add(Activation('relu', name='conv_2_act'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu', name='dense_1'))
model.add(Dense(units=10, activation='softmax', name='dense_2'))

model.build(input_shape=(None, 28, 28, 1))
model.summary()

#print(model.layers)
#print(model.layers[0].get_weights())  # 특정 layer의 trainable parameter의 값을 가져올 수 있다.
# %%