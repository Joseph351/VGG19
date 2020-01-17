# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:55:10 2019

@author: Joseph Duodu
"""

import os as os
import numpy as np

class data:
    def __init__(self):
        pass
        
    def generate_data(self, file_Path):
        train_records = sorted(os.listdir(file_Path))[1:1025]
        val_records = sorted(os.listdir(file_Path))[1025:1154]
        train_paths = []
        val_paths = []
        for i in train_records:
            train_paths.append(file_Path+i)
        for i in val_records:
            val_paths.append(file_Path+i)    
        return train_paths, val_paths
    
    def normalize_data(self, X_train, X_test):
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean)/(std +1e-7)
        X_test = (X_test-mean)/(std +1e-7)
        return X_train, X_test