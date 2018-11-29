#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:42:11 2018

@author: pohsuanh


Fully Covolutional Network FCN-32s. 

FCN-32s network is based on VGG-16

"""

import os
import tensorflow as tf
import numpy as np
from sklearn.datasets import laod_sample_images

root_dir = '/home/pohsuanh/Documents/Computer_Vision/HW6'

os.chdir(root_dir)

img_path = os.path.join(root_dir, 'HW6_Dataset','image')

train_imgs = os.path.join(img_path, 'train')

test_imgs = os.pathjoin(img_path, 'test')

height = 224

width =  224

channels = 3
X = tf.placeholder(shape = (None, height, width, channels))

