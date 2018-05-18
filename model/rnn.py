import tensorflow as tf
import numpy as np

def biLSTM(input_tensor):
    """
    The bidirectional lstm part of the model
    the input_tensor size should be (batch_size, 99, num_cnn_filter(default is 64))

    """
    # Prepare data shape to match 'bidirectional_rnn' function requirements
    # current data input shape: (batch_size, 99, 64)
    # needed: 99 * [(batch_size, 64)], a 1-dim list with len 99
    N, T, D = [i.value for i in input_tensor.shape]
    hidden_dim = D//2
    input_tensor = tf.unstack(input_tensor, T, axis=1)

    # define lstm cells, let the hidden cell num is D
    with tf.variable_scope('rnn'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, name='lstm_fw_cell')
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, name='lstm_bw_cell')
        try:
            rnn_outputs, _, _ = tf.nn.static_bidirectional_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=input_tensor,
                dtype=tf.float32,
                scope='rnn'
            )
        except: # for old tf version
            rnn_outputs = tf.nn.static_bidirectional_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=input_tensor,
                dtype=tf.float32,
                scope='rnn'
            )
        stack_output = tf.stack(rnn_outputs, axis=1)
        print(stack_output.shape)
        dim = np.prod([i.value for i in stack_output.shape[1:]])
        flatten_output = tf.reshape(stack_output, shape=[-1, dim])
    return flatten_output


