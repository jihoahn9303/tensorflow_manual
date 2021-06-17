# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:30:52 2021

@author: jiho Ahn
@topic: binary classification
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

plt.style.use('seaborn')

n_sample = 50
x_train = np.random.normal(0, 1, size=(n_sample, 1)).astype(np.float32)
y_train = (x_train >= 0).astype(np.float32)

#fig, ax = plt.subplots(figsize=(10, 10))
#ax.scatter(x_train, y_train)
#ax.tick_params(labelsize=20)

class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='sigmoid')
        
    def call(self, x):
        prediction = self.d1(x)
        return prediction

EPOCHS = 10
LR = 0.01    
model = Classifier()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

loss_metric = tf.keras.metrics.Mean()  # calculate mean of loss
acc_metric = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1, 1))
        y = tf.reshape(y, (1, 1))
        
        with tf.GradientTape() as tape:
            predictions = model(x)  # forward propagation
            loss = loss_object(y, predictions)  # calculate loss
            
        gradients = tape.gradient(loss, model.trainable_variables)  # back propagation
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # update parameters
        
        loss_metric(loss)  # append loss value 
        acc_metric(y, predictions)  # append (label, pred) set
        
    print(colored("Epoch: ", 'cyan', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%'
    
    ds_loss = loss_metric.result()  # calculate mean of total loss
    ds_acc = acc_metric.result()  # calculate accracy of total prediction
    
    print(template.format(ds_loss, ds_acc*100))
    
    loss_metric.reset_states()
    acc_metric.reset_states()
       