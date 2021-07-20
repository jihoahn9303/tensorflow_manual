# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:32:24 2021

@author: jiho Ahn
@topic: Model Configuration
"""
from termcolor import colored

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

# build
model.build(input_shape=(None, 28, 28, 1))

print(colored("model.layers ", 'cyan'), '\n', model.layers, '\n')
print(colored("len(model.layers) ", 'cyan'), '\n', len(model.layers), '\n')

# %%

final_layer = model.layers[-1]
final_layer_config = final_layer.get_config()

print(json.dumps(final_layer_config, indent=2))

# %%

for layer in model.layers:
    layer_config = layer.get_config() 
    
    layer_name = layer_config['name']
    if layer_name.startswith('conv') and len(layer_name.split('_')) <= 2:
        print(colored('Layer name: ', 'cyan'), layer_name)
        print('n filters: ', layer_config['filters'])
        print('kernel size: ', layer_config['kernel_size'])
        print('padding: ', layer_config['padding'])
        print()
        
    if layer_name.endswith('act'):
        print(colored('Layer name: ', 'cyan'), layer_name)
        print('activation: ', layer_config['activation'])
        print()
        
        
# %%

final_layer = model.layers[-1]

type(final_layer.get_weights())

print(colored("type(final_layer.get_weights())", 'cyan'), '\n',
      type(final_layer.get_weights()), '\n')
print(colored("type(final_layer.get_weights()[0])", 'cyan'), '\n',
      type(final_layer.get_weights()[0]), '\n')
print(colored("type(final_layer.get_weights()[1])", 'cyan'), '\n',
      type(final_layer.get_weights()[1]), '\n')
print(colored("final_layer.get_weights()[0].shape", 'cyan'), '\n',
      final_layer.get_weights()[0].shape, '\n')
print(colored("final_layer.get_weights()[1].shape", 'cyan'), '\n',
      final_layer.get_weights()[1].shape, '\n')


# %%

















