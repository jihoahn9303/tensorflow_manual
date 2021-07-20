# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:12:18 2021

@author: jiho Ahn
@topic: load dataset with tensorflow(2) 
        (tensorflow_datasets)
"""

import tensorflow_datasets as tfds

# load dataset with additional information

dataset, ds_info = tfds.load(name='mnist', 
                             shuffle_files=True, 
                             with_info=True)


print(ds_info)
print(ds_info.features)
print(ds_info.splits)


# %%
'''
dataset = tfds.load(name='mnist', 
                    shuffle_files=True)

train_ds = dataset['train'].batch(32)
test_ds = dataset['test'] 

for tmp in train_ds:
    
    print(type(tmp))
    print(tmp.keys())
    
    images = tmp['image']
    labels = tmp['label']
    
    print(images.shape)
    print(labels.shape)
    break
    
'''

# %%
# auto-unpack dataset to images and labels
'''
dataset = tfds.load(name='mnist',
                    shuffle_files=True,
                    as_supervised=True)

train_ds = dataset['train'].batch(32)
test_ds = dataset['test']


for tmp in train_ds:
    images = tmp[0]
    labels = tmp[1]
    
    print(images.shape)
    print(labels.shape)
    
    break


for images, labels in train_ds:
    print(images.shape)
    print(labels.shape)
    
    break
'''

# %%
# auto-split data(1)
'''
(train_ds, test_ds), ds_info = tfds.load(name='mnist',
                                         shuffle_files=True,
                                         as_supervised=True,
                                         split=['train', 'test'],
                                         with_info=True)

train_ds = train_ds.batch(32)

for images, labels in train_ds:
    print(images.shape)
    print(labels.shape)
    break
'''    
# %%
# auto-split data(2)
(train_ds, validation_ds, test_ds), ds_info = tfds.load(name='patch_camelyon',
                                                        shuffle_files=True,
                                                        as_supervised=True,
                                                        split=['train', 'validation', 'test'], 
                                                        with_info=True,
                                                        batch_size=9)

# train_ds = train_ds.batch(9)
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

import matplotlib.pyplot as plt
    
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15), squeeze=False)

for ax_idx, ax in enumerate(axes.flat):
    ax.imshow(images[ax_idx, ...].numpy())
    ax.set_title(labels[ax_idx].numpy(), fontsize=30)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
