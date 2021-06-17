# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:28:45 2021

@author: jiho Ahn
@topic: validation / test, tf.function decorator
"""
from termcolor import colored
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

n_train, n_validation, n_test = 1000, 300, 300

train_x = np.random.normal(0, 1, size=(n_train, 1)).astype(np.int32)
train_x_noise = train_x + np.random.normal(0, 1, size=(n_train, 1)).astype(np.int32)
train_y = (train_x_noise > 0).astype(np.int32)

# create validation set
validation_x = np.random.normal(0, 1, size=(n_validation, 1)).astype(np.int32)
validation_x_noise = validation_x + np.random.normal(0, 1, size=(n_validation, 1)).astype(np.int32)
validation_y = (validation_x_noise > 0).astype(np.int32)

# create test set
test_x = np.random.normal(0, 1, size=(n_test, 1)).astype(np.int32)
test_x_noise = test_x + np.random.normal(0, 1, size=(n_test, 1)).astype(np.int32)
test_y = (test_x_noise > 0).astype(np.int32)

# %%
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(n_train).batch(8)

# create batch set for validation
validation_ds = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
validation_ds = validation_ds.batch(n_validation)

# create batch set for testing
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(n_test)

# %%
# phase 1: hard coding(set model ~ test model)
model = Sequential()
model.add(Dense(units=2, activation='softmax'))

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

validation_loss = Mean()
validation_acc = SparseCategoricalAccuracy()

test_loss = Mean()
test_acc = SparseCategoricalAccuracy()

'''
EPOCHS = 10
train_losses, train_accs = [], []
validation_losses, validation_accs = [], []
for epoch in range(EPOCHS):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(y, predictions)
        
    for x, y in validation_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)
        
        validation_loss(loss)
        validation_acc(y, predictions)
        
    print(colored('Epoch: ', 'red', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n' +\
                'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(), train_acc.result()*100,
                          validation_loss.result(), validation_acc.result()*100))
        
    train_losses.append(train_loss.result())
    train_accs.append(train_acc.result()*100)
    validation_losses.append(validation_loss.result())
    validation_accs.append(validation_acc.result()*100)
    
    train_loss.reset_states()
    train_acc.reset_states()
    validation_loss.reset_states()
    validation_acc.reset_states()
    
for x, y in test_ds:
    predictions = model(x)
    loss = loss_object(y, predictions)
    
    test_loss(loss)
    test_acc(y, predictions)
    
print(colored('Final result: ', 'red', 'on_white'))
template = 'Test Loss: {:.4f}\t Test Accuracy: {:.2f}%\n' 
print(template.format(test_loss.result(), test_acc.result()*100))

test_loss.reset_states()
test_acc.reset_states()
'''

# %%
# phase 2: set model ~ test model with making function and using @tf.function decorator
@tf.function
def train_step(x, y):
    global model, loss_object
    global train_loss, train_acc
    global optimizer
    
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(y, predictions)
    

@tf.function    
def validation():
    global validation_ds, model, loss_object
    global validation_loss, validation_acc
       
    for x, y in validation_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)
        
        validation_loss(loss)
        validation_acc(y, predictions)
        
        
def train_reporter():
    global epoch
    global train_loss, train_acc
    global validation_loss, validation_acc
    
    print(colored('Epoch: ', 'red', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n' +\
                'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(), train_acc.result()*100,
                          validation_loss.result(), validation_acc.result()*100))


def metric_resetter():
    global train_loss, train_acc
    global validation_loss, validation_acc

    train_losses.append(train_loss.result())
    train_accs.append(train_acc.result()*100)
    validation_losses.append(validation_loss.result())
    validation_accs.append(validation_acc.result()*100)
    
    train_loss.reset_states()
    train_acc.reset_states()
    validation_loss.reset_states()
    validation_acc.reset_states()
    
    
EPOCHS = 10
train_losses, train_accs = [], []
validation_losses, validation_accs = [], []   
 
for epoch in range(EPOCHS):
    for x, y in train_ds:
        train_step(x, y)
        
    validation()
    train_reporter()
    metric_resetter()
    
for x, y in test_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)
    
        test_loss(loss)
        test_acc(y, predictions)
    
print(colored('Final result: ', 'cyan', 'on_white'))
template = "Test Loss: {:.4f}\t Test Accuracy: {:.2f}%\n"
print(template.format(test_loss.result(), test_acc.result()*100))
    
