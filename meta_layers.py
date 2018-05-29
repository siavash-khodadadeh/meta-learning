import tensorflow as tf


def conv3d(input_layer, weights, bias, strides=None, padding='VALID', activation=None, name=None):
    with tf.variable_scope(name):
        if strides is None:
            strides = (1, 1)

        strides = (1, ) + strides + (1, )
        conv3d_out = tf.nn.conv3d(input_layer, weights, strides, padding=padding)

        if activation is not None:
            return activation(conv3d_out + bias)

        return conv3d_out + bias


def conv2d(input_layer, weights, bias, strides=None, padding='VALID', activation=None, name=None):
    with tf.variable_scope(name):
        if strides is None:
            strides = (1, 1)

        strides = (1,) + strides + (1, )
        conv2d_out = tf.nn.conv2d(input_layer, weights, strides, padding=padding)

        if activation is not None:
            return activation(conv2d_out) + bias

        return conv2d_out + bias


def dense(input_layer, weights, bias, activation, name=None):
    with tf.variable_scope(name):
        dense_out = tf.matmul(input_layer, weights) + bias
        if activation is not None:
            return activation(dense_out)
        return dense_out
