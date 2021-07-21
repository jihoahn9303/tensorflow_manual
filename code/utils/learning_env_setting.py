# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:39:30 2021

@author: jiho Ahn
@topic: Utility Functions(1) - env setting
"""

'''
--Structure of project directory 
    -- main.py
    -- models.py
    -- utils
        -- basic_utils.py
        -- cp_utils.py
        -- dataset_utils.py
        -- learning_env_setting.py
        -- train_validation_test.py
    
    -- train folder 
        -- confusion_matrix
        -- model
        -- losses_accs.npz
        -- losses_accs.visualization.png
        -- test_result.txt
'''

import os 
import shutil
from datetime import datetime
from termcolor import colored

import numpy as np
import tensorflow as tf

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

dir_name = 'train_' + datetime.now().strftime('%Y_%m_%d')
#CONTINUE_LEARNING = False  

def dir_setting(dir_name, CONTINUE_LEARNING):    
    '''
    Make or delete train directory 
    
    Input: 1) dir_name: name for train directory which contains model information
           2) CONTINUE_LEARNING: Flag for learning status.
                                 If you set this value False, it means you want to discard result of training
                                 or it's the first time to train the model.
                                 If you set this values True, it means you want to train the latest model.
                        
    Output: Dictionary which contains path for result of training information
                                 
    '''
    cp_path = os.path.join(os.getcwd(), dir_name)
    confusion_path = os.path.join(cp_path, 'confusion_matrix')
    model_path = os.path.join(cp_path, 'model')
    
    if (CONTINUE_LEARNING == False) and (os.path.isdir(cp_path)):
        shutil.rmtree(cp_path)
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)
        os.makedirs(confusion_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        
    path_dict = {
                'cp_path': cp_path,
                'confusion_path': confusion_path,
                'model_path': model_path
            }
    return path_dict


def get_classification_metrics():
    '''
    Return metric objects for trained model.
    
    Output: loss and accuracy object for train/validation/test
    '''
    train_loss = Mean()
    train_acc = SparseCategoricalAccuracy()
    
    validation_loss = Mean()
    validation_acc = SparseCategoricalAccuracy()
    
    test_loss = Mean()
    test_acc = SparseCategoricalAccuracy()
    
    metric_objects = dict()
    metric_objects['train_loss'] = train_loss
    metric_objects['train_acc'] = train_acc
    metric_objects['validation_loss'] = validation_loss
    metric_objects['validation_acc'] = validation_acc
    metric_objects['test_loss'] = test_loss
    metric_objects['test_acc'] = test_acc
    
    return metric_objects


def continue_setting(CONTINUE_LEARNING, path_dict, model=None):
    '''
    Load the latest model information to continue training model.
    
    Input: 1) CONTINUE_LEARNING: Flag for learning status.
                                 If you set this value False, it means that it's the first time to train the model.
                                 If you set this values True, it means you want to train the latest model.
           2) path_dict: Dictionary which contains path of model information
           3) model: deep learning model that you want to train
    
    Output: 1) model: the latest model in the directory or new model that you want to train
            2) losses_accs: Dictionary which contains metric information
            3) start_epoch: the latest training epoch 
    '''

    if (CONTINUE_LEARNING == True) and (len(os.listdir(path_dict['model_path'])) == 0):
        CONTINUE_LEARNING = False
        print(colored('CONTINUE LEARNING flag has been converted to FALSE', 'cyan'))
        
    if CONTINUE_LEARNING == True:
        epoch_list = os.listdir(path_dict['model_path'])
        epoch_list = [int(epoch.split('_')[1]) for epoch in epoch_list]
        last_epoch = epoch_list.sort()[-1]
        last_model_path = path_dict['model_path'] + '/epoch_' + str(last_epoch)
        
        model = tf.keras.models.load_model(last_model_path)
        
        losses_accs_path = path_dict['cp_path']
        losses_accs_np = np.load(losses_accs_path + '/losses_accs.npz')
        losses_accs_dict = dict()
        
        for k, v in losses_accs_np.items():
            losses_accs_dict[k] = list(v)
            
        start_epoch = last_epoch + 1
        
    else:
        model = model
        start_epoch = 0
        losses_accs = {'train_losses': [], 'train_accs': [],
                       'validation_losses': [], 'validation_accs': []}
    
    return model, losses_accs, start_epoch 


#dir_setting(dir_name, False)
# %%
    
