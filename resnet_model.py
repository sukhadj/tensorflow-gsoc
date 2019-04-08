"""Contains definitions for Residual Networks.
Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
The script only implemented only for resnet version 1 with bottleneck
"""

from __future__ import  absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

# This implementation uses keras functional API
# Fixed padding

# batch normalization does not have training parameter in keras API
def batch_norm(training, data_format):
    return layers.BatchNormalization(axis= 1 if data_format=='channels_first' else 3,
                                    momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON,
                                    center = True, scale=True,
                                    fused = True)

def conv2d_fixed_padding(filters, kernel_size, strides, data_format):

    return layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=('SAME' if strides == 1 else 'VALID'),
                            use_bias=False, kernel_initializer=tf.keras.initializers.VarianceScaling(),
                            data_format=data_format)

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides,
                        data_format):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(training=training, data_format=data_format)(shortcut)

    inputs = conv2d_fixed_padding(filters=filters, kernel_size=3, strides=strides,
                                    data_format=data_format)(inputs)

    inputs =  batch_norm(training=training, data_format=data_format)(inputs)

    inputs = tf.keras.activations.relu(inputs)

    inputs = conv2d_fixed_padding(filters=filters, kernel_size=3, strides=strides,
                                    data_format=data_format)(inputs)(inputs)

    inputs = batch_norm(training=training, data_format=data_format)(inputs)

    inputs += shortcut
    inputs = tf.keras.activations.relu(inputs)

    return inputs

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    filters_out = filters * 4 if bottleneck else filters
    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                     data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)

class Model(object):
    """Base class for building the Resnet Model"""
    def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
                kernel_size,
                conv_stride, first_pool_size, first_pool_stride,
                block_size, block_stride,
                resnet_version = DEFAULT_VERSION, data_format = None,
                dtype = DEFAULT_DTYPE):

        self.resnet_size = resnet_size

        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda()
            else 'channels_last')

        self.resnet_version = resnet_version

        if resnet_version not in (1,2):
            raise ValueError('Resnet version should be 1 or 2')

        self.bottleneck = bottleneck

        if bottleneck:
            if resnet_version == 1 :
                self.block_fn = _bottleneck_block_v1
            else :
                pass;
        else:
            pass;

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                                *args, **kwargs):
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        # check
        return tf.compat.v1.variable_scope('resnet_model',
                                       custom_getter=self._custom_dtype_getter)

    def __class__(self, inputs, training):
        if self.data_format == 'channels_first':
            inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
             filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)(inputs)

        inputs = tf.identity(inputs, 'initial_conv')

        if self.resnet_version == 1:
            inputs = batch_norm(training, self.data_format)(inputs)
            inputs = tf.keras.activations.relu(inputs)

        if self.first_pool_size:
            inputs = layers.MaxPooling2D(
                pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)(inputs)

            inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2**i)
            inputs = block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, blocks=num_blocks,
                strides=self.block_strides[i], training=training,
                name='block_layer{}'.format(i + 1), data_format=self.data_format)

        if self.pre_activation:
            inputs = batch_norm(training, self.data_format)(inputs)
            inputs = tf.keras.activations.relu(inputs)

        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
        inputs = tf.identity(inputs, 'final_reduce_mean')

        inputs = tf.squeeze(inputs, axes)
        inputs = tf.layers.Dense(units=self.num_classes)(inputs)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs
