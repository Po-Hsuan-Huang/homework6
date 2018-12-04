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
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)


# assign each run to a separate log file, so the tensorboard can function properly. 
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

root_logdir = "logs"

logdir = "{}/run-{}/".format(root_logdir,now)

def fcn_model_fn(features, labels, mode):
    
    L2 = tf.contrib.layers.l2_regularizer(scale=0.1)
    
    trainable = False
    
    if  mode == tf.estimator.ModeKeys.TRAIN  :
        
        trainable = True
    
    seed = 2019
    
    with tf.name_scope("vgg16_pretrained"):
        
        x = tf.layers.conv2d(features, 64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv1_1',
                          kernel_regularizer= L2,
                          trainable = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp1_1')
    
        x =  tf.layers.conv2d(x, 64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv1_2',
                          kernel_regularizer= L2,
                          trainable  = trainable)
     
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp1_2')
    
        x =  tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='pool1')
        
        # Block 2
        x = tf.layers.conv2d(x, 128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv2_1',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp2_1')
    
        
        x = tf.layers.conv2d(x, 128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv2-2',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp2_2')
    
    
        x = tf.layers.max_pooling2d(x,(2, 2), strides=(2, 2), name='pool2')
        
        # Block 3
        x = tf.layers.conv2d (x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv3_1',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp3_1')
    
        x = tf.layers.conv2d (x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv3_2',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp3_2')
    
        
        x = tf.layers.conv2d (x, 256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv3_3',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp3_3')
    
        
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='pool3')
        
        # Block 4
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv4_1',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp4_1')
    
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv4_2',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp4_2')
    
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv4_3',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp4_3')
    
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='pool4')
        
        # Block 5
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv5_1',
                          kernel_regularizer= L2,
                          trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp5_1')
        
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv5_2',
                          kernel_regularizer= L2,
                          trainable  = trainable)
    
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp5_2')
        
        x = tf.layers.conv2d (x, 512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='conv5_3',
                          kernel_regularizer= L2,
                          trainable  = trainable)
    
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp5_3')
    
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='pool5')
    
    with tf.name_scope("deconv_layers"):
        # Block 6
        
        x = tf.layers.conv2d(x, 4096, (7,7), 
                             activation='relu',
                             padding='same',
                             name='conv6_1',
                             kernel_regularizer= L2,
                             trainable  = trainable)
        
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp6_1')
    
        x = tf.layers.conv2d(x, 4096, (1,1),
                             activation='relu',
                             padding='same',
                             name='conv6_2',
                             kernel_regularizer= L2,
                             trainable  = trainable)
    
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp6_2')
        
        x = tf.layers.conv2d(x, 1, (1,1),
                             activation='relu',
                             padding='same',
                             name='conv6_3',
                             kernel_regularizer= L2,
                             trainable  = trainable) 
    
        x = tf.layers.dropout(x, rate = 0.4, seed = seed, training = trainable , name ='dp6_3')
        
        # There are two classes [1: road, 0:  non-road]
        heatmap = tf.layers.conv2d_transpose(x, 1, (64,64), strides=(32,32),
                                       activation='linear',
                                       padding='same',
                                       name='deconv6_1',
                                       kernel_regularizer= L2,
                                       trainable  = trainable)
        
        logit = tf.nn.sigmoid(heatmap, name = 'logit')
        
        pred = tf.to_int32(logit > 0.5)
        
        pred = tf.squeeze(pred, axis = 3)

#    print(heatmap.shape)
    
    # Do pixel-wise classification :

    predictions = {
            
      # Generate predictions (for PREDICT and EVAL mode)
      
      "classes": pred, # tf.argmax(logit, axis =3 )
      
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the logging_hook`.
      
      "probabilities": logit  #tf.nn.softmax(logit, name="softmax_tensor")

      }
    

    if mode == tf.estimator.ModeKeys.PREDICT:
      
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    # Homework requires tf.nn.sigmoid_cross_entropy_with_logits()
    if False : 
        # ignore where label is -1 , which corresponds to Void.
        
        logit_f = tf.reshape(heatmap, (-1,1,1,1)) # flatten the output
            
        logit_f = tf.squeeze(logit_f, axis = [2,3])
        
        label_f = tf.reshape(labels,(-1,1))
        
        keep = tf.where(tf.greater_equal(labels, 0) )
        
        logit_f = tf.gather(logit_f, keep)
        
        label_f = tf.gather(label_f, keep)
        
        tf.assert_equal(tf.shape(label_f)[0], tf.shape(logit_f)[0])
        
        tf.assert_non_negative(label_f) # Void is labelled -1, which should be excluded from the loss func
    
    
        # sigmoid_cross_entorpy implements tf.nn.sparse_signoid_cross_entropy_with_logit, 
        # it will convert output to logit in the op
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = label_f, logits=logit_f)
    
    heatmap = tf.squeeze(heatmap, axis =3)

    label_f = tf.to_int32(labels > 0)

    tf.assert_equal(tf.shape(label_f), tf.shape(heatmap))

    tf.assert_non_negative(label_f)

    loss = tf.losses.sigmoid_cross_entropy( multi_class_labels = label_f ,logits = heatmap)    
    # Configure the trainable Op (for TRAIN mode)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
    
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum = 0.99)
        
        train_op = optimizer.minimize(loss=loss, global_step = tf.train.get_global_step())
        
        tf.summary.scalar('train_loss', loss)
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    
    # Set up logging for metrics
    
    iou = tf.metrics.mean_iou(label_f,predictions['classes'], num_classes = 2 , name = 'mean_iou')
         
    eval_metric_ops = {"IoU": iou}

    tensors_to_log_prob = {"probabilities": "deconv_layers/logit"}
    
    tensors_to_log_iou = {"mean_iou": iou}
    
    tf.summary.scalar('mean_iou', iou[0])

    logging_hook = tf.train.LoggingTensorHook(
                                   tensors=tensors_to_log_iou, every_n_iter=200)
    
    if mode == tf.estimator.ModeKeys.EVAL :
        
        tf.summary.scalar('eval_loss', loss)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops = eval_metric_ops)

    
#%%
if __name__ == "__main__":
    
    root_dir = '/home/pohsuanh/Documents/Computer_Vision/HW6'

    # Load training and eval data
  
    train_data, eval_data, test_data, gt = data_load.load()
    
    # Flags
    
    TRAIN = False

    PREDICT = True    

    DRAW_SAMPLE = False
    
    # Construct model
    if DRAW_SAMPLE == True :

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
    
    pretrained_weights = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=os.path.join(root_dir,'pretrained_weights','vgg_16.ckpt'),
            vars_to_warm_start= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'vgg16_pretrained'))
    
    fcn_segmentor = tf.estimator.Estimator(
    
    model_fn=fcn_model_fn, model_dir=os.path.join(root_dir, 'ckpts'),  warm_start_from= pretrained_weights) 
        
    if TRAIN == True :
    
        for epoch in range(100):
        
        # Train the model
        
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=train_data['x'],
                y=train_data['y'],
                batch_size=1,
                num_epochs=None, # number of epochs to iterate over data. If None will run forever.
                shuffle=True)
           
            fcn_segmentor.train(
                input_fn=train_input_fn,
                steps=200
                )
           
        # Evaluate the model and print results
           
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=eval_data['x'],
                y=eval_data['y'],
                num_epochs=1,
                batch_size=10,
                shuffle=False)
           
            eval_results = fcn_segmentor.evaluate(input_fn=eval_input_fn)
           
            print('eval_loss :', eval_results)
        
    
    
#%%  We withhold the predction from test set untill all the hyperparameters are finetuned.
    
    if PREDICT == True :
        
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=test_data['x'],
                y=test_data['y'],
                batch_size =1,
                num_epochs=1,
                shuffle=False)
        
        # predict method returns a generator
        
        pred = list( fcn_segmentor.predict(input_fn = pred_input_fn))
        
        pred = [p['classes'] for p in pred]
            
        fig = plt.figure(1, figsize=(32,16))
            
        for i, p in enumerate(pred) : 
            
            fig.add_subplot(3,1,1)
            
            plt.title('camera photo')
            
            plt.imshow(test_data['x'][i])
            
            fig.add_subplot(3,1,2)
            
            plt.title('prediction')
            
            plt.imshow(p)
            
            fig.add_subplot(3,1,3)
            
            plt.title('ground truth')
            
            plt.imshow(gt['test'][i])
            
            filename = 'pred_{}.png'.format(i)
            
            plt.savefig(os.path.join(root_dir,'predictions',filename))