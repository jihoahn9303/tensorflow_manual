# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:03:37 2021

@author: jiho Ahn
@topic: LeNet Implementation
"""
import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, ZeroPadding2D

# %%
# Implementation for LeNet1 with Sequential
LeNet1 = Sequential()
LeNet1.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet1.add(AveragePooling2D(pool_size=2, strides=2))
LeNet1.add(Conv2D(filters=12, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet1.add(AveragePooling2D(pool_size=2, strides=2))

LeNet1.add(Flatten())
LeNet1.add(Dense(units=10, activation='softmax'))

LeNet1.build(input_shape=(None, 28, 28, 1))
LeNet1.summary()

# %%
# Implementation for LeNet4 with Sequential
LeNet4 = Sequential()
LeNet4.add(ZeroPadding2D(padding=2))
LeNet4.add(Conv2D(filters=4, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet4.add(AveragePooling2D(pool_size=2, strides=2))
LeNet4.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet4.add(AveragePooling2D(pool_size=2, strides=2))

LeNet4.add(Flatten())
LeNet4.add(Dense(units=120, activation='tanh'))
LeNet4.add(Dense(units=10, activation='softmax'))

LeNet4.build(input_shape=(None, 28, 28, 1))
LeNet4.summary()

# %%
# Implementation for LeNet5 with Sequential
LeNet5 = Sequential()
LeNet5.add(ZeroPadding2D(padding=2))
LeNet5.add(Conv2D(filters=6, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet5.add(AveragePooling2D(pool_size=2, strides=2))
LeNet5.add(Conv2D(filters=16, kernel_size=5, padding='valid', strides=1,
                  activation='tanh'))
LeNet5.add(AveragePooling2D(pool_size=2, strides=2))

LeNet5.add(Flatten())
LeNet5.add(Dense(units=140, activation='tanh'))
LeNet5.add(Dense(units=84, activation='tanh'))
LeNet5.add(Dense(units=10, activation='softmax'))

LeNet5.build(input_shape=(None, 28, 28, 1))
LeNet5.summary()

# %%
# Implementation for LeNet1 with sub-classing
class FeatureExtractor(Layer):
    def __init__(self, filter1, filter2):
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = Conv2D(filters=filter1, kernel_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)
        
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        return x
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
                    'conv1_num_filters' : self.conv1.get_config()['filters'],
                    'conv1_num_kernel_size' : self.conv1.get_config()['kernel_size'],
                    'conv1_padding' : self.conv1.get_config()['padding'],
                    'conv1_strides' : self.conv1.get_config()['strides'],
                    'conv1_activation' : self.conv1.get_config()['activation'],
                    'conv1_pool_size' : self.conv1_pool.get_config()['pool_size'],
                    'conv1_pool_strides' : self.conv1_pool.get_config()['strides'],
                    'conv2_num_filters' : self.conv2.get_config()['filters'],
                    'conv2_num_kernel_size' : self.conv2.get_config()['kernel_size'],
                    'conv2_padding' : self.conv2.get_config()['padding'],
                    'conv2_strides' : self.conv2.get_config()['strides'],
                    'conv2_activation' : self.conv2.get_config()['activation'],
                    'conv2_pool_size' : self.conv2_pool.get_config()['pool_size'],
                    'conv2_pool_strides' : self.conv2_pool.get_config()['strides']
                })
#        config.update({
#                    'conv1' : self.conv1.get_config(),
#                    'conv1_pool' : self.conv1_pool.get_config(),
#                    'conv2' : self.conv2.get_config(),
#                    'conv2_pool' : self.conv2_pool.get_config(),
#                })
    
        return config
    
    
class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__() 
        
        # feature extractor
        self.fe = FeatureExtractor(4, 12) 

        
        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=10, activation='softmax')) 
        
        
    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)
        return x
    
    
class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__() 
        
        # feature extractor        
        self.zero_padding = ZeroPadding2D(padding=2)
        self.fe = FeatureExtractor(4, 16)
               
        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=120, activation='tanh')) 
        self.classifier.add(Dense(units=10, activation='softmax')) 
        
               
    def call(self, x):
        x = self.zero_padding(x)
        x = self.fe(x)
        x = self.classifier(x)
        return x
        
        
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__() 
        
        # feature extractor
        self.zero_padding = ZeroPadding2D(padding=2)
        self.fe = FeatureExtractor(6, 16)
        
        # classifier
        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(units=140, activation='tanh')) 
        self.classifier.add(Dense(units=84, activation='tanh')) 
        self.classifier.add(Dense(units=10, activation='softmax'))
        
              
    def call(self, x):
        x = self.zero_padding(x)
        x = self.fe(x)
        x = self.classifier(x)
        return x

# %%
lenet5 = LeNet5()
lenet5.build(input_shape=(None, 28, 28, 1))
lenet5.summary()
# print(lenet5.layers[1].get_config())
# %%



























