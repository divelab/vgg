import tensorflow as tf


def conv2d(inputs, num_outputs, kernel_size, scope):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        activation_fn=None, biases_initializer=None)
    outputs = batch_norm(outputs, scope)
    return outputs


def pool2d(inputs, kernel_size, scope):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME')


def dense(inputs, dim, scope):
    outputs = tf.contrib.layers.fully_connected(
        inputs, dim, scope=scope+'/dense')
    outputs = batch_norm(outputs, scope)
    return outputs


def batch_norm(inputs, scope):
    return tf.contrib.layers.batch_norm(
        inputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm')
