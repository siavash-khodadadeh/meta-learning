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


def batch_normalization(input_layer, mean, variance, offset, name=None):
    # from tensorflow.python.ops import math_ops
    # from tensorflow.python.ops import state_ops
    # from tensorflow.python.framework import ops

    # def do_update(variable, value, momentum):
    #     with ops.name_scope(None, 'AssignMovingAvg', [variable, value, momentum]) as scope:
    #         with ops.colocate_with(variable):
    #             decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
    #             if decay.dtype != variable.dtype.base_dtype:
    #                 decay = math_ops.cast(decay, variable.dtype.base_dtype)
    #             update_delta = (variable - value) * decay
    #             return state_ops.assign_sub(variable, update_delta, name=scope)

    with tf.variable_scope(name):
        # new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        # new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
        #
        # do_update(mean, new_mean, momentum=0.999)
        # do_update(variance, new_variance, momentum=0.999)
        return tf.nn.batch_normalization(input_layer, mean, variance, offset, scale=None, variance_epsilon=1e-3)
