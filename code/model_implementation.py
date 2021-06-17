# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:35:19 2021

@author: jiho Ahn
"""

"""
Methods of model implementation
1. Sequential
2. Functional
3. Model Sub Classing
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# ex) Simple linear regression with Sequential

# 1. generate normal distribution data
x_train = tf.random.normal(shape=(1000,), dtype=tf.float32)
y_train = 3 * x_train + 1 + 0.2*tf.random.normal(shape=(1000,), dtype=tf.float32)

x_test = tf.random.normal(shape=(300,), dtype=tf.float32)
y_test = 3 * x_test + 1 + 0.2*tf.random.normal(shape=(300,), dtype=tf.float32)

# 2. set model
model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1, activation='linear')
        ])

# 3. compile model(add loss function and optimizer)
model.compile(loss='mean_squared_error', optimizer='SGD')

# 4. train model
model.fit(x_train, y_train, epochs=50, verbose=2)

# 5. test model
model.evaluate(x_test, y_test, verbose=2)

# %%
# ex) Simple linear regression with model subclassing
from termcolor import colored
import tensorflow as tf

# 1. generate normal distribution data
x_train = tf.random.normal(shape=(1000,), dtype=tf.float32)
y_train = 3 * x_train + 1 + 0.2*tf.random.normal(shape=(1000,), dtype=tf.float32)

x_test = tf.random.normal(shape=(300,), dtype=tf.float32)
y_test = 3 * x_test + 1 + 0.2*tf.random.normal(shape=(300,), dtype=tf.float32)

# 2. define class
class LinearPredictor(tf.keras.Model):
    # initialization with gaining tf.keras.model 
    def __init__(self):
        super(LinearPredictor, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='linear')
    
    # define how to propagate values(forward propagation)    
    def call(self, x):
        x = self.d1(x)
        return x

# 3. generate model instance
LR = 0.01    
EPOCHS = 10
model = LinearPredictor()

# 4. generate loss object instance and optimizer instance
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

# 5. learn model
for epoch in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1, 1))
        # logs calculation result on the tape
        with tf.GradientTape() as tape:
            predictions = model(x)  # LinearPredictor.call(x) -> prediction
            loss = loss_object(y, predictions) 
        
        # model.trainable_variables -> contain weight and bias info
        # back propagation
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # update weight and bias parameters
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)
    
    template = 'Train Loss: {:.4f}\n'
    print(template.format(loss))