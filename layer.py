import numpy as np
import tensorflow as tf


def weight_variable(shape, mean=0.0, stddev=0.02, name=None):
    return tf.get_variable(
        name+"/weight",
        shape=shape,
        # initializer=tf.contrib.layers.xavier_initializer(),
        initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev),
        dtype=tf.float32
    )


def bias_variable(shape, name=None):
    return tf.get_variable(
        name+"/bias",
        shape=shape,
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32
    )


def conv2d(x, W, stride=None, padding=None):
    stride = stride or [1, 1, 1, 1]
    padding = padding or 'SAME'
    return tf.nn.conv2d(x, W, strides=stride, padding=padding)


def batch_normalization(x, is_training, scope, pop_mean=None, pop_var=None, epsilon=1e-5, decay=0.9):
    with tf.variable_scope(scope):
        shape = x.get_shape()
        size = shape.as_list()[-1]
        axis = list(range(len(shape)-1))

        scale = tf.get_variable('scale', [size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        offset = tf.get_variable('offset', [size], initializer=tf.constant_initializer(0.0))

        # device 2
        if pop_mean is None and pop_var is None:
            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(tf.float32), trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(tf.float32), trainable=False)

        batch_mean, batch_var = tf.nn.moments(x, axis)

        def batch_statistics():
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(is_training, batch_statistics, population_statistics)


def instance_norm(x, scope, is_training=None, epsilon=1e-5, decay=0.9):
    with tf.variable_scope(scope):
        shape = x.get_shape()
        size = shape.as_list()[-1]
        axis = list(range(len(shape)-1))[1:]

        scale = tf.get_variable('scale', [size], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [size], initializer=tf.constant_initializer(0.0))

        batch_mean, batch_var = tf.nn.moments(x, axis)
        return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)


def lrelu(x, neg_thres=0.2):
    y = tf.maximum(x, neg_thres*x)
    return y


def conv_block(
        input_t,
        output_channel,
        name,
        kernel_size,
        stride_size=1,
        padding='SAME',
        stddev=0.02,
        activation=None,
        norm=None,
        is_training=None,
):
    input_channel = input_t.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w_conv = weight_variable([kernel_size, kernel_size, input_channel, output_channel], stddev=stddev, name=name)
        output_t = conv2d(input_t, w_conv, stride=[1, stride_size, stride_size, 1], padding=padding)
        b_conv = bias_variable([output_channel], name=name)
        output_t = tf.nn.bias_add(output_t, b_conv)
        if norm is not None:
            output_t = norm(x=output_t, is_training=is_training, scope="normalization")

        if activation is not None:
            output_t = activation(output_t)

    return output_t


def deconv_block(
        input_t,
        output_shape,
        layer_name,
        stride_size=2,
        kernel_size=4,
        padding='SAME',
        stddev=0.02,
        activation=None,
        norm=None,
        is_training=None,
):
    inpus_channels = input_t.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        w_conv = weight_variable([kernel_size, kernel_size, output_shape[-1], inpus_channels], stddev=stddev, name=layer_name)
        output_t = tf.nn.conv2d_transpose(input_t, w_conv, output_shape, [1, stride_size, stride_size, 1], padding=padding)
        b_conv = bias_variable([output_shape[-1]], name=layer_name)
        output_t = tf.nn.bias_add(output_t, b_conv)

        if norm is not None:
            output_t = norm(output_t, is_training=is_training, scope='normalization')

        if activation is not None:
            output_t = activation(output_t)

    return output_t


def res_block(
        input_t,
        output_shape,
        layer_name,
):
    output_t = input_t

    with tf.variable_scope(layer_name):
        output_t = tf.pad(output_t, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        output_t = conv_block(output_t, output_shape, "c1", kernel_size=3, norm=instance_norm, padding='VALID', activation=tf.nn.relu)
        output_t = tf.pad(output_t, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        output_t = conv_block(output_t, output_shape, "c2", kernel_size=3, norm=instance_norm, padding='VALID')
        output_t = output_t + input_t
    return output_t

