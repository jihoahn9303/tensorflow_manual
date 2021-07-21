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
    
    -- train1 (매 학습마다)
        -- confusion_matrix
        -- model
        -- losses_accs.npz
        -- losses_accs.visualization.png
        -- test_result.txt
'''

import os
from utils.learning_env_setting import dir_setting
from datetime import datetime
from termcolor import colored

import numpy as np
import tensorflow as tf


dir_name = 'train_' + datetime.now().strftime('%Y_%m_%d')
CONTINUE_LEARNING = True

path_dict = dir_setting(dir_name, CONTINUE_LEARNING)

model = 'test'

def continue_setting(CONTINUE_LEARNING, path_dict, model=None):

    if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0:
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
