import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from model.param_init import kaiming_normal

def get_cnn_model(input_tensor, reuse=False):
    """
    Get the CNN part of the model, the structure of the CNN is:
        conv(64, 4*4, stride=1, valid) -> relu -> [not use conv(64, 1*4, stride=1, valid) -> relu] -> max_pooling(2) -> faltten
    TODO: add regularizer
    :param input_tensor: input tensor of CNN with shape (N, 4, 300, 1) (N, H, W, C)
    :Return flatten_tensor: the input of next part of model
    """
    N, H, W, C = [i.value for i in input_tensor.shape]
    N = input_tensor.shape[0]
    conv_size = 4
    #regularizer = layers.l2_regularizer()
    with tf.variable_scope('cnn', reuse=reuse):
        conv_w1 = tf.get_variable(
            'conv_w1', 
            #shape=[conv_size, conv_size, C, 64], 
            initializer=kaiming_normal((conv_size, conv_size, C, 64)),
            dtype=tf.float32
            )
        conv_b1 = tf.get_variable('conv_b1', initializer=tf.zeros([64]), dtype=tf.float32)
        # conv_b1 = tf.zeros(shape=[64] ,name='conv_b1', dtype=tf.float32) 
        activation = tf.nn.conv2d(input_tensor, conv_w1, strides=(1,1,1,1), padding='VALID', name='conv_activation1') + conv_b1 #(1*297*64)
        relu_out = tf.nn.relu(activation, name='conv_relu1') 
        pooling_out = tf.nn.max_pool(relu_out, ksize=(1,1,3,1), strides=(1,1,3,1), padding='VALID', name='conv_pool1') #(1*99*64)
        cnn_out = tf.squeeze(pooling_out, axis=1)
        #new_dim = pooling_out.shape[-1].value * pooling_out.shape[-2].value * pooling_out.shape[-3].value
        #cnn_result = tf.reshape(pooling_out, [-1, new_dim]) # (N, 99*64)
    return cnn_out






