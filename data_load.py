#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:30:32 2018

@author: pohsuanh
"""
import os
import numpy as np
from matplotlib.image import imread
from glob import glob as glob
import pickle
import tensorflow as tf

def load():
        
    root_dir = '/home/pohsuanh/Documents/Computer_Vision/HW6'
    
    os.chdir(root_dir)


    if 'train_data' not in locals():
    
        img_path = os.path.join(root_dir, 'HW6_Dataset','image')
        
        label_path = os.path.join(root_dir, 'HW6_Dataset', 'label')
        
        train_imgs = sorted( glob(os.path.join(img_path, 'train', '*png')))
        
        train_labels = sorted( glob(os.path.join(label_path,'train', '*label')))
        
        test_imgs = sorted( glob(os.path.join(img_path, 'test', '*png')))
        
        test_labels = sorted( glob(os.path.join(label_path, 'test', '*label')))
        
    val_set  = 45
    
    
    # return data structure of (N, H, W, C) 
    # both label and image are 4d-tensor of same shape
    # in the semantic label, 0 means non-road, 1 means road, -1 means void
    # remember to ignore -1 when computing the loss function.
    
#    train_data = {'x' : tf.convert_to_tensor( np.asarray([ imread(i) for i in train_imgs[:-val_set] ] )),
#                  'y': tf.convert_to_tensor( np.asarray([pickle.load(open(i,'rb'),encoding='latin1')  for i in train_labels[:-val_set] ]))
#                  }
#    
#    eval_data = {'x' : tf.convert_to_tensor(np.asarray([ imread(i) for i in train_imgs[val_set:]])),
#                 'y': tf.convert_to_tensor(np.asarray([ pickle.load(open(i,'rb'),encoding='latin1') for i in train_labels[val_set:]]))
#                  }
#    
#    test_data = {'x' : tf.convert_to_tensor(np.asarray([ imread(i) for i in test_imgs])),
#                 'y' : tf.convert_to_tensor(np.asarray([ pickle.load(open(i,'rb'),encoding='latin1')  for i in test_labels]))   
#                  }
    train_data = {'x' : np.asarray([ imread(i) for i in train_imgs[:-val_set] ] ),
                  'y':  np.asarray([pickle.load(open(i,'rb'),encoding='latin1')  for i in train_labels[:-val_set] ])
                  }
    
    eval_data = {'x' : np.asarray([ imread(i) for i in train_imgs[val_set:]]),
                 'y': np.asarray([ pickle.load(open(i,'rb'),encoding='latin1') for i in train_labels[val_set:]])
                  }
    
    test_data = {'x' : np.asarray([ imread(i) for i in test_imgs]),
                 'y' : np.asarray([ pickle.load(open(i,'rb'),encoding='latin1')  for i in test_labels])   
                  }

    
    return train_data, eval_data, test_data

 