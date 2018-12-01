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
import matplotlib.pyplot as plt
import data_load


def fcn_model_fn(features, labels, mode):
    
    L2 = tf.contrib.layers.l2_regularizer(scale=0.1)
    
    trainable = False
    
    if  mode == tf.estimator.ModeKeys.TRAIN  :
        
        trainable = True
    
    seed = 2019
    
    x = tf.layers.conv2d(features, 64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1',
                      kernel_regularizer= L2,
                      trainable = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block1_dp1')

    x =  tf.layers.conv2d(x, 64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2',
                      kernel_regularizer= L2,
                      trainable  = trainable)
 
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block2_dp2')

    x =  tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block1_pool')
    
    # Block 2
    x = tf.layers.conv2d(x, 128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block2_dp1')

    
    x = tf.layers.conv2d(x, 128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block2_dp2')


    x = tf.layers.max_pooling2d(x,(2, 2), strides=(2, 2), name='block2_pool')
    
    # Block 3
    x = tf.layers.conv2d (x, 256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block3_dp1')

    x = tf.layers.conv2d (x, 256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block3_dp2')

    
    x = tf.layers.conv2d (x, 256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block3_dp3')

    
    x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block3_pool')
    
    # Block 4
    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block4_dp1')

    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block4_dp2')

    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block4_dp3')

    x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block4_pool')
    
    # Block 5
    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1',
                      kernel_regularizer= L2,
                      trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block5_dp1')
    
    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2',
                      kernel_regularizer= L2,
                      trainable  = trainable)

    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block5_dp2')
    
    x = tf.layers.conv2d (x, 512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3',
                      kernel_regularizer= L2,
                      trainable  = trainable)

    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block5_dp3')

    x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block5_pool')
    
    # Block 6
    
    x = tf.layers.conv2d(x, 4096, (7,7), 
                         activation='relu',
                         padding='same',
                         name='block6_conv1',
                         kernel_regularizer= L2,
                         trainable  = trainable)
    
    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block6_dp1')

    x = tf.layers.conv2d(x, 4096, (1,1),
                         activation='relu',
                         padding='same',
                         name='block6_conv2',
                         kernel_regularizer= L2,
                         trainable  = trainable)

    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block6_dp2')
    
    x = tf.layers.conv2d(x, 1, (1,1),
                         activation='relu',
                         padding='same',
                         name='block6_conv3',
                         kernel_regularizer= L2,
                         trainable  = trainable) 

    x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='block6_dp3')
    
    # There are two classes [road, non-road]
    logit = tf.layers.conv2d_transpose(x, 2, (64,64), strides=(32,32),
                                   activation='sigmoid',
                                   padding='same',
                                   name='block6_deconv1',
                                   kernel_regularizer= L2,
                                   trainable  = trainable)

    print(logit.shape)
    # Do pixel-wise classification :

    predictions = {
            
      # Generate predictions (for PREDICT and EVAL mode)
      
      "classes": tf.argmax(logit, axis =3 ),
      
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the logging_hook`.
      
      "probabilities": tf.nn.softmax(logit, name="softmax_tensor")

      }

    if mode == tf.estimator.ModeKeys.PREDICT:
      
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    # Homework requires tf.nn.sigmoid_cross_entropy_with_logits()
    
    # ignore where label is -1 , which corresponds to Void.
    
    logit_f = tf.reshape(logit, (-1,1,1,1)) 
        
    logit_f = tf.squeeze(logit_f, axis = [2,3])
    
    label_f = tf.reshape(labels,(-1,1))
    
    keep = tf.where(tf.greater_equal(labels, 0) )
    
    logit_f = tf.gather(logit_f, keep)
    
    label_f = tf.gather(label_f, keep)
    
    tf.assert_equal(tf.shape(label_f)[0], tf.shape(logit_f)[0])
    
    tf.assert_non_negative(label_f) # Void is labelled -1, which should be excluded from the loss func

    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(label_f,tf.int32), logits=logit_f)
    
    # Configure the trainable Op (for TRAIN mode)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
    
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.000001, momentum = 0.99)
        
        train_op = optimizer.minimize(loss=loss, global_step = tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    
    tp = tf.metrics.true_positives(labels,predictions['classes'])
    
    fp = tf.metrics.false_positives(labels,predictions['classes'])
    
    fn = tf.metrics.false_negatives(labels,predictions['classes'])
    
    eval_metric_ops = {"IoU": tp/(tp + fp + fn)}
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    
    root_dir = '/home/pohsuanh/Documents/Computer_Vision/HW6'

    # Load training and eval data
  
    train_data, eval_data, test_data = data_load.load()
    
    # Construct model
#    pic = np.random.randint((test_data['x']).shape[0])
    pic = np.random.randint(len(test_data['x']))

    image_sample = test_data['x'][pic]
    
    label_sample = test_data['y'][pic]
    
#    image_sample = tf.Session().run(image_sample)
#    
#    label_sample = tf.Session().run(label_sample)
    
    plt.figure(figsize=(20,40))
    plt.title('data')
    plt.imshow(image_sample)
    
    plt.figure(figsize =(20,40))
    plt.title('gt')
    plt.imshow(label_sample)
        
    # Create the Estimator
    
    fcn_segmentor = tf.estimator.Estimator(
    
    model_fn=fcn_model_fn, model_dir=root_dir)
   
    # Set up logging for predictions

    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
                                   tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data['x'],
        y=train_data['y'],
        batch_size=1,
        num_epochs=None, # number of epochs to iterate over data. If None will run forever.
        shuffle=True)
   
    fcn_segmentor.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
   
    # Evaluate the model and print results
   
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data['x'],
        y=eval_data['y'],
        num_epochs=1,
        shuffle=False)
   
    eval_results = fcn_segmentor.evaluate(input_fn=eval_input_fn)
   
    print(eval_results)


