import tensorflow as tf

if tf.__version__ < '2.3.0':
    from tensorflow.python.keras import layers
else:
    from tensorflow.python.keras.layers import VersionAwareLayers

    layers = VersionAwareLayers()

from tensorflow.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.engine import training

import math
import copy

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 12,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 24,
    'filters_out': 32,
    'expand_ratio': 12,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.
}, ]


def get_attention_adjusted(x, num_timesteps, prefix, permute=True, dropout=0., add_noise=0.):
    """

    :param x:
    :return:
    """

    if permute:
        x = layers.Permute((2, 1))(x)  # Output: batch_size x nb_filters x timestep

    if add_noise > 0.:
        x = layers.GaussianNoise(stddev=add_noise)(x)

    wgt = layers.Dense(units=num_timesteps, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                       name=prefix + 'att_0')(x)
    wgt = layers.Dense(units=num_timesteps, activation='sigmoid',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-5), name=prefix + 'att_1')(wgt)
    x = layers.Multiply()([x, wgt])  # Output: batch_size x nb_filters x timestep

    x = layers.Permute((2, 1))(x)  # Output: batch_size x timestep x nb_filters

    x = layers.GlobalAveragePooling1D()(x)  # Output: batch_size x nb_filter
    if dropout > 0.:
        x = layers.GaussianDropout(dropout)(x)
    return x


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
    """An inverted residual block.
    Arguments:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'expand_conv')(
            inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding='same',
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_reduce')(
            se)
        se = layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x


def low_connection(inputs, num_filters, activation, bn_axis, prefix, postfix):
    x = layers.Conv2D(
        num_filters,
        3,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix + 'top_conv' + postfix)(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'top_bn' + postfix)(x)
    x = layers.Activation(activation, name=prefix + 'top_activation' + postfix)(x)

    x = layers.GlobalAveragePooling2D(name=prefix + 'avg_pool' + postfix)(x)

    return x


def low_feat_concat(inputs, num_features=0):
    """

    :param inputs: list
    :param num_features:
    :return:
    """

    if num_features == 0:
        x = tf.stack(inputs, axis=2)
        x = get_attention_adjusted(x, num_timesteps=len(inputs), permute=False, dropout=0., add_noise=0., prefix='ulf')
    else:
        x = tf.concat(inputs, axis=1)
        x = layers.Dense(num_features, kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='low_high_concat_dense')(x)
        x = layers.BatchNormalization(name='low_high_concat_bn')(x)
        x = layers.Activation('relu', name='low_high_concat_act')(x)

    return x


def AFCNet(width_coefficient,
           depth_coefficient,
           default_size=64,
           drop_connect_rate=0.,
           depth_divisor=8,
           activation='swish',
           blocks_args='default',
           model_name='AFCNet',
           weights=None,
           input_tensor=None,
           input_shape=None,
           pooling='avg',
           augmentation=None, rescale=False, prefix='',
           num_top_filters=32, use_low_feat=False):
    """
    Modified from Efficient
    https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/applications/efficientnet.py
    """
    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name=prefix[:-1])
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

        # Build stem

    x = img_input
    if len(input_shape) > 3:
        x = tf.reshape(x, (-1, 48, 48, 3))

    if augmentation is not None:
        x = augmentation(x)
    if rescale:
        x = layers.Rescaling(1. / 255.)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, 3),
        name=prefix + 'stem_conv_pad')(x)
    x = layers.Conv2D(
        round_filters(32),
        3,
        strides=2,
        padding='valid',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix + 'stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'stem_bn')(x)
    x = layers.Activation(activation, name=prefix + 'stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    low_feats = []
    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name=prefix + 'block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1

        if use_low_feat:
            low_feats.append(low_connection(x, round_filters(num_top_filters), activation, bn_axis, prefix, postfix=str(i)))
    # low_connection(x, round_filters(num_top_filters), activation, bn_axis, prefix, postfix=0)
    # Build top
    x = layers.Conv2D(
        round_filters(num_top_filters),
        3,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=prefix + 'top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=prefix + 'top_bn')(x)
    x = layers.Activation(activation, name=prefix + 'top_activation')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name=prefix + 'avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name=prefix + 'max_pool')(x)

    if use_low_feat:
        x = low_feat_concat([x, ] + low_feats, num_features=0)  # num_features > 0 => flatten, else avg

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    return model
