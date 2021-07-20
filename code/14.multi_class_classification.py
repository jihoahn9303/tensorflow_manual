# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:49:20 2021

@author: jiho Ahn
@topic: Multi-class Classification
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from termcolor import colored

np.random.seed(1)
plt.style.use('seaborn')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

### dataset generation
n_class, n_data = 10, 500
noise = 0.05

x_data = np.empty(shape=(0, 2))
y_data = np.empty(shape=(0, 1))

cmap = cm.get_cmap('rainbow', lut=n_class)
flg, ax = plt.subplots(figsize=(10, 10))

for class_idx in range(n_class):
    center = np.random.uniform(-1, 1, (2, ))
    
    x1_data = center[0] + noise*np.random.normal(0, 1, (n_data, 1))
    x2_data = center[1] + noise*np.random.normal(0, 1, (n_data, 1))
    
    class_x_data = np.hstack((x1_data, x2_data))
    class_y_data = class_idx*np.ones((n_data, 1))
    
    x_data = np.vstack((x_data, class_x_data)).astype(np.float32)
    y_data = np.vstack((y_data, class_y_data)).astype(np.int32)
    
    ax.scatter(x1_data, x2_data, color=cmap(class_idx),
               alpha=0.3)
    
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])

# %%
### Tensorflow dataset
import tensorflow as tf

train_ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
train_ds = train_ds.shuffle(1000).batch(8)

### set model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(units=32))
model.add(Activation('relu'))
model.add(Dense(units=n_class))
model.add(Activation('softmax'))

### instantiate training object
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

loss_object = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=0.01)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

### define trainer
@tf.function
def trainer():
    global train_ds, model, loss_object, optimizer
    global train_loss, train_acc
    
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        train_loss(loss)
        train_acc(y, predictions)
        
        
### define reporter
def training_reporter():
    global epoch
    global train_loss, train_acc  

    print(colored('Epoch', 'red', 'on_white'), epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%'
    print(template.format(train_loss.result(),
                          train_acc.result()*100))
        
    train_loss.reset_states()
    train_acc.reset_states()
    

### training
EPOCHS = 50

for epoch in range(EPOCHS):
    trainer()
    training_reporter()
    
# %%
# make decision boundary
    
x1_test = np.linspace(-1.2, 1.2, 100).astype(np.float32)
x2_test = np.linspace(-1.2, 1.2, 100).astype(np.float32)

X1, X2 = np.meshgrid(x1_test, x2_test)
x1_test, x2_test = X1.flatten(), X2.flatten()

x_test = np.hstack((x1_test.reshape(-1, 1),
                    x2_test.reshape(-1, 1)))

#print(x_test.shape)
y_test = model(x_test).numpy()  # 예측 확률
predictions = np.argmax(y_test, axis=1) # 예측 클래스

cdict = {i:cmap(i) for i in range(n_class)}
color_arr = [cdict[pred] for pred in predictions]

fig, ax = plt.subplots(figsize=(10, 10))
for class_idx in range(n_class):
    # scatter train set
    ax.scatter(x_data[class_idx*n_data : (class_idx+1)*n_data, 0],
               x_data[class_idx*n_data : (class_idx+1)*n_data, 1],
               c=cmap(class_idx))

# plot decision boundary    
ax.scatter(x1_test, x2_test, c=color_arr, alpha=0.2)


