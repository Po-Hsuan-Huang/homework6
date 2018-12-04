# homework6
run FCN-32.py 


# current progress : 

1. overfitting

The model can  predict IoU, the Loss of training set is decreasing, while val_set loss stagnates.

2. unable to visualize tensorboard during trianing/ evaluation

The log file cannot be read by tensorbaord --logdir=./, due to ther are more than one event per run. I am not sure what that means.
By change I am able to log in the tensorboard after the process being terminated or existed. tf.Summary.scaler() are correctly recorded. However, I did not see tensor graph in tensorboard.

3. Warm_restart from pretrained vgg16 weights doesn't work due to variable name confusion. 

The pretrained weights is defined with tf.nn.{op}, while my model is based on high-level wraper tf.estimator(input_model_fn). All of my layers are designated a name and scope. I don't know if the varaiable name of pretrained weights and layers in my model should be the same 

4. predict output not implemented yet.

The final result should be exhibited as an image which can be comapred with images in HW6_dataset/gt folder.
