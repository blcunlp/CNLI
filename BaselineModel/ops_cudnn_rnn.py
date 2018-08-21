'''
forked from https://github.com/baidu-research/GloballyNormalizedReader/blob/master/ops.py
'''

import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from itertools import zip_longest
import queue
import threading
import numpy as np

#######cudnn_lstm##########
def cudnn_lstm(inputs, num_layers, hidden_size, is_training, direction='bidirectional',regularizer=None,scope=None):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:  Number of units in each layer.
        - direction: indicate 'bidirectional' or 'unidirectional'     
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])

    cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers, hidden_size, input_size,
        input_mode="linear_input", direction=direction)

    est_size = estimate_cudnn_lstm_parameter_size(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode="linear_input",
        direction=direction)

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer(),
        regularizer=regularizer)

    num_dir = direction_to_num_directions(direction)
    # initial_state: a tuple of tensor(s) of shape`[num_layers * num_dirs, batch_size, num_units]
    init_state = tf.tile(
        tf.zeros([num_dir * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])  # [num_dir * num_layers, batch_size, hidden_size]
    '''
    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor(s) of shape
        `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
        zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
    '''
    hiddens, output_h, output_c = cudnn_cell(
        inputs,
        input_h=init_state,
        input_c=init_state,
        params=cudnn_params,
        is_training=True)

    # Convert to batch major
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    output_h = tf.transpose(output_h, [1, 0, 2])
    output_c = tf.transpose(output_c, [1, 0, 2])

    #return hiddens, output_h, output_c
    return hiddens

#######cudnn_gru##########

def cudnn_gru(inputs, num_layers, hidden_size, is_training, direction='bidirectional',scope=None):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:  Number of units in each layer.
        - direction: indicate 'bidirectional' or 'unidirectional'     
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    ref: https://github.com/tensorflow/tensorflow/issues/13860
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])
   
    cudnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(
        num_layers, hidden_size, input_size,
        input_mode="linear_input", direction=direction)

    est_size = estimate_cudnn_gru_parameter_size(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode="linear_input",
        direction=direction)

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer())

    num_dir = direction_to_num_directions(direction)
    # initial_state: a tuple of tensor(s) of shape`[num_layers * num_dirs, batch_size, num_units]
    init_state = tf.tile(
        tf.zeros([num_dir * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])  # [num_dir * num_layers, batch_size, hidden_size]
    '''
    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor(s) of shape
        `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
        zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
    '''
    #hiddens, output_h, output_c = cudnn_cell(
    hiddens, output_h = cudnn_cell(
        inputs,
        input_h=init_state,
        params=cudnn_params,
        is_training=True)

    # Convert to batch major
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    output_h = tf.transpose(output_h, [1, 0, 2])
    #output_c = tf.transpose(output_c, [1, 0, 2])

    #return hiddens,  output_h
    return hiddens

def estimate_cudnn_lstm_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params

def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def estimate_cudnn_gru_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_gru_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params
 

def cudnn_gru_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 6 * hidden_size
    weights = 3 * (hidden_size * input_size) + 3 * (hidden_size * hidden_size)
    return biases + weights               

def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))

def parameter_count():
    """Return the total number of parameters in all Tensorflow-defined
    variables, using `tf.trainable_variables()` to get the list of
    variables."""
    return sum(np.product(var.get_shape().as_list())
               for var in tf.trainable_variables())
