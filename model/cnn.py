import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from param_init import kaiming_normal

def get_cnn_model(input_tensor):
    """
    Get the CNN part of the model, the structure of the CNN is:
        conv(64, 4*4, stride=1, valid) -> relu -> [not use conv(64, 1*4, stride=1, valid) -> relu] -> max_pooling(2) -> faltten
    TODO: add regularizer
    :param input_tensor: input tensor of CNN with shape (N, 4, 300, 1) (N, H, W, C)
    :Return flatten_tensor: the input of next part of model
    """
    N, H, W, C = input_tensor.shape
    conv_size = 4
    pooling_size = 2

    #regularizer = layers.l2_regularizer()
    with tf.variable_scope('cnn'):
        conv_w1 = tf.get_variable(
            'conv_w1', 
            shape=[conv_size, conv_size, C, 64], 
            initializer=kaiming_normal((conv_size, conv_size, C, 64)),
            dtype=tf.float64
            )
        conv_b1 = tf.zeros(shape=[64] ,name='conv_b1', dtype=tf.float64) 
        activation = tf.nn.conv2d(input_tensor, conv_w1, strides=(1,1,1,1), padding='VALID') + conv_b1 #(1*297)
        relu_out = tf.nn.relu(activation) 
        pooling_out = tf.nn.max_pool(relu_out, ksize=(1,1,3,1), strides=(1,1,3,1), padding='VALID') #(1*99)






