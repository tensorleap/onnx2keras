import logging
from functools import partial
from typing import List

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from .utils import ensure_tf_type, is_numpy
from .tfops_funcs import tf_transpose, tf_pad, tf_shape, tf_reshape


def _normalize_output_padding(output_padding, rank):
    if not output_padding:
        return (0,) * rank
    if len(output_padding) == rank:
        return tuple(output_padding)
    raise AttributeError('Invalid output_padding for GroupedConvTranspose')


def _normalize_pads(pads, rank):
    if not pads:
        return (0,) * (2 * rank)
    if len(pads) == 2 * rank:
        return tuple(pads)
    if len(pads) == rank:
        return tuple(pads) + tuple(pads)
    raise AttributeError('Invalid pads for GroupedConvTranspose')


def _conv2d_transpose_valid(x, k, strides, dilations, output_padding):
    in_shape = tf.shape(x)  # [N, H, W, Cin]
    n = in_shape[0]
    h = in_shape[1]
    w = in_shape[2]
    k_h = tf.shape(k)[0]
    k_w = tf.shape(k)[1]
    out_ch = tf.shape(k)[2]

    s_h, s_w = strides
    d_h, d_w = dilations
    op_h, op_w = _normalize_output_padding(output_padding, 2)

    out_h = s_h * (h - 1) + ((k_h - 1) * d_h + 1) + op_h
    out_w = s_w * (w - 1) + ((k_w - 1) * d_w + 1) + op_w
    output_shape = tf.stack([n, out_h, out_w, out_ch])

    return tf.nn.conv2d_transpose(
        input=x,
        filters=k,
        output_shape=output_shape,
        strides=[1, s_h, s_w, 1],
        padding="VALID",
        data_format="NHWC",
        dilations=[1, d_h, d_w, 1],
    )


def _conv3d_transpose_valid(x, k, strides, dilations, output_padding):
    in_shape = tf.shape(x)  # [N, D, H, W, Cin]
    n = in_shape[0]
    d = in_shape[1]
    h = in_shape[2]
    w = in_shape[3]
    k_d = tf.shape(k)[0]
    k_h = tf.shape(k)[1]
    k_w = tf.shape(k)[2]
    out_ch = tf.shape(k)[3]

    s_d, s_h, s_w = strides
    d_d, d_h, d_w = dilations
    op_d, op_h, op_w = _normalize_output_padding(output_padding, 3)

    out_d = s_d * (d - 1) + ((k_d - 1) * d_d + 1) + op_d
    out_h = s_h * (h - 1) + ((k_h - 1) * d_h + 1) + op_h
    out_w = s_w * (w - 1) + ((k_w - 1) * d_w + 1) + op_w
    output_shape = tf.stack([n, out_d, out_h, out_w, out_ch])

    return tf.nn.conv3d_transpose(
        input=x,
        filters=k,
        output_shape=output_shape,
        strides=[1, s_d, s_h, s_w, 1],
        padding="VALID",
        data_format="NDHWC",
        dilations=[1, d_d, d_h, d_w, 1],
    )


def _crop_pads_2d(y, pads):
    h0, w0, h1, w1 = _normalize_pads(pads, 2)
    if h0 or w0 or h1 or w1:
        sh = tf.shape(y)[1]
        sw = tf.shape(y)[2]
        y = y[:, h0:sh - h1, w0:sw - w1, :]
    return y


def _crop_pads_3d(y, pads):
    d0, h0, w0, d1, h1, w1 = _normalize_pads(pads, 3)
    if d0 or h0 or w0 or d1 or h1 or w1:
        sd = tf.shape(y)[1]
        sh = tf.shape(y)[2]
        sw = tf.shape(y)[3]
        y = y[:, d0:sd - d1, h0:sh - h1, w0:sw - w1, :]
    return y


def grouped_conv_transpose(x, kernel=None, bias=None, kernel_shape=None, bias_shape=None, strides=(1, 1),
                           dilations=(1, 1), pads=None, output_padding=None, groups=1,
                           data_format="channels_first", trainable_weights=False, name=None):
    strides = tuple(strides)
    dilations = tuple(dilations)
    pads = tuple(pads) if pads is not None else ()
    output_padding = tuple(output_padding) if output_padding is not None else ()
    groups = int(groups)

    if kernel_shape is None and kernel is not None:
        kernel_shape = tuple(kernel.shape)
    if bias_shape is None and bias is not None:
        bias_shape = tuple(bias.shape)

    if kernel_shape is None:
        raise AttributeError('GroupedConvTranspose requires kernel_shape')

    def target_layer(x_in):
        def _normalize_output_padding_local(out_pad, rank):
            if not out_pad:
                return (0,) * rank
            if len(out_pad) == rank:
                return tuple(out_pad)
            raise AttributeError('Invalid output_padding for GroupedConvTranspose')

        def _normalize_pads_local(pads_val, rank):
            if not pads_val:
                return (0,) * (2 * rank)
            if len(pads_val) == 2 * rank:
                return tuple(pads_val)
            if len(pads_val) == rank:
                return tuple(pads_val) + tuple(pads_val)
            raise AttributeError('Invalid pads for GroupedConvTranspose')

        def _conv2d_transpose_valid_local(xv, kv):
            in_shape = tf.shape(xv)  # [N, H, W, Cin]
            n = in_shape[0]
            h = in_shape[1]
            w = in_shape[2]
            k_h = tf.shape(kv)[0]
            k_w = tf.shape(kv)[1]
            out_ch = tf.shape(kv)[2]

            s_h, s_w = strides
            d_h, d_w = dilations
            op_h, op_w = _normalize_output_padding_local(output_padding, 2)

            out_h = s_h * (h - 1) + ((k_h - 1) * d_h + 1) + op_h
            out_w = s_w * (w - 1) + ((k_w - 1) * d_w + 1) + op_w
            output_shape = tf.stack([n, out_h, out_w, out_ch])

            return tf.nn.conv2d_transpose(
                input=xv,
                filters=kv,
                output_shape=output_shape,
                strides=[1, s_h, s_w, 1],
                padding="VALID",
                data_format="NHWC",
                dilations=[1, d_h, d_w, 1],
            )

        def _conv3d_transpose_valid_local(xv, kv):
            in_shape = tf.shape(xv)  # [N, D, H, W, Cin]
            n = in_shape[0]
            d = in_shape[1]
            h = in_shape[2]
            w = in_shape[3]
            k_d = tf.shape(kv)[0]
            k_h = tf.shape(kv)[1]
            k_w = tf.shape(kv)[2]
            out_ch = tf.shape(kv)[3]

            s_d, s_h, s_w = strides
            d_d, d_h, d_w = dilations
            op_d, op_h, op_w = _normalize_output_padding_local(output_padding, 3)

            out_d = s_d * (d - 1) + ((k_d - 1) * d_d + 1) + op_d
            out_h = s_h * (h - 1) + ((k_h - 1) * d_h + 1) + op_h
            out_w = s_w * (w - 1) + ((k_w - 1) * d_w + 1) + op_w
            output_shape = tf.stack([n, out_d, out_h, out_w, out_ch])

            return tf.nn.conv3d_transpose(
                input=xv,
                filters=kv,
                output_shape=output_shape,
                strides=[1, s_d, s_h, s_w, 1],
                padding="VALID",
                data_format="NDHWC",
                dilations=[1, d_d, d_h, d_w, 1],
            )

        def _crop_pads_2d_local(yv):
            h0, w0, h1, w1 = _normalize_pads_local(pads, 2)
            if h0 or w0 or h1 or w1:
                sh = tf.shape(yv)[1]
                sw = tf.shape(yv)[2]
                yv = yv[:, h0:sh - h1, w0:sw - w1, :]
            return yv

        def _crop_pads_3d_local(yv):
            d0, h0, w0, d1, h1, w1 = _normalize_pads_local(pads, 3)
            if d0 or h0 or w0 or d1 or h1 or w1:
                sd = tf.shape(yv)[1]
                sh = tf.shape(yv)[2]
                sw = tf.shape(yv)[3]
                yv = yv[:, d0:sd - d1, h0:sh - h1, w0:sw - w1, :]
            return yv

        dtype = x_in.dtype
        if kernel is None:
            kernel_t = tf.zeros(kernel_shape, dtype=dtype)
        else:
            kernel_t = tf.convert_to_tensor(kernel)
            if kernel_t.dtype != dtype:
                kernel_t = tf.cast(kernel_t, dtype)
            if trainable_weights:
                kernel_t = tf.Variable(kernel_t, trainable=True,
                                       name=None if name is None else f"{name}_kernel")

        if bias_shape is not None:
            if bias is None:
                bias_t = tf.zeros(bias_shape, dtype=dtype)
            else:
                bias_t = tf.convert_to_tensor(bias)
                if bias_t.dtype != dtype:
                    bias_t = tf.cast(bias_t, dtype)
                if trainable_weights:
                    bias_t = tf.Variable(bias_t, trainable=True,
                                         name=None if name is None else f"{name}_bias")
        else:
            bias_t = None

        rank = kernel_t.shape.rank - 2
        if rank == 2:
            if data_format == "channels_first":
                x_in = tf.transpose(x_in, [0, 2, 3, 1])
            if groups == 1:
                y = _conv2d_transpose_valid_local(x_in, kernel_t)
            else:
                x_splits = tf.split(x_in, num_or_size_splits=groups, axis=-1)
                k_splits = tf.split(kernel_t, num_or_size_splits=groups, axis=-1)
                ys = [_conv2d_transpose_valid_local(xi, ki)
                      for xi, ki in zip(x_splits, k_splits)]
                y = tf.concat(ys, axis=-1)
            y = _crop_pads_2d_local(y)
            if bias_t is not None:
                y = y + tf.reshape(bias_t, [1, 1, 1, -1])
            if data_format == "channels_first":
                y = tf.transpose(y, [0, 3, 1, 2])
            return y
        if rank == 3:
            if data_format == "channels_first":
                x_in = tf.transpose(x_in, [0, 2, 3, 4, 1])
            if groups == 1:
                y = _conv3d_transpose_valid_local(x_in, kernel_t)
            else:
                x_splits = tf.split(x_in, num_or_size_splits=groups, axis=-1)
                k_splits = tf.split(kernel_t, num_or_size_splits=groups, axis=-1)
                ys = [_conv3d_transpose_valid_local(xi, ki)
                      for xi, ki in zip(x_splits, k_splits)]
                y = tf.concat(ys, axis=-1)
            y = _crop_pads_3d_local(y)
            if bias_t is not None:
                y = y + tf.reshape(bias_t, [1, 1, 1, 1, -1])
            if data_format == "channels_first":
                y = tf.transpose(y, [0, 4, 1, 2, 3])
            return y
        raise AttributeError('GroupedConvTranspose supports only 2D/3D')

    return keras.layers.Lambda(target_layer, name=name)(x)

def calculate_permute_values(n_dims: int, to_channel_first: bool) -> List[int]:
    if to_channel_first:
        return [n_dims - 1] + list(range(1, n_dims - 1))
    else:
        return list(range(2, n_dims)) + [1]


def permute_wrap_conv_if_constant(partial_func, conv_input, is_constant, conv_channels, params, weights=None):
    if is_constant:
        input_shape = tf_shape(conv_input, tf_name=f"{params['cleaned_name']}_conv_wrap_shape")
        permuted = keras.layers.Permute(calculate_permute_values(len(input_shape), to_channel_first=False),
                                        name=f"{params['cleaned_name']}_conv_wrap_permute_1")(conv_input)
        conv_layer = partial_func(data_format="channels_last")
        conv_res = conv_layer(permuted)
        if weights is not None:
            try:
                conv_layer.set_weights(weights)
            except ValueError:
                expected_shapes = [w.shape for w in conv_layer.get_weights()]
                reshaped = [w.reshape(es) if w.shape != es else w for w, es in zip(weights, expected_shapes)]
                conv_layer.set_weights(reshaped)
        result = keras.layers.Permute(calculate_permute_values(len(input_shape), to_channel_first=True),
                                      name=f"{params['cleaned_name']}_conv_wrap_permute_2")(conv_res)
    else:
        data_fmt = keras.backend.image_data_format()
        conv = partial_func(data_format=data_fmt)
        if data_fmt == 'channels_first':
            channels_idx = 1
        else:
            channels_idx = -1
        # Determine correct input channels from weights if available
        expected_in_channels = conv_channels
        if weights is not None:
            grp = getattr(conv, 'groups', 1)
            expected_in_channels = weights[0].shape[-2] * grp

        if conv_input.shape[channels_idx] is None or conv_input.shape[channels_idx] != expected_in_channels:
            # Reshape input to match expected channels (dynamic shape or shape mismatch)
            conv_input_shape = tf_shape(conv_input, tf_name=f"{params['cleaned_name']}_conv_wrap_shape_1")
            conv_input = tf_reshape(conv_input, [*conv_input_shape[:channels_idx], expected_in_channels,
                                                 *conv_input_shape[channels_idx + 1:]],
                                    tf_name=f"{params['cleaned_name']}_conv_wrap_reshape_2")
        result = conv(conv_input)
        if weights is not None:
            conv.set_weights(weights)
    return result


def convert_conv(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.conv')

    if len(node.input) == 3:
        logger.debug('Conv with bias')
        # Has bias
        has_bias = True
        W = layers[node.input[1]]
        bias = layers[node.input[2]]

    elif len(node.input) == 2:
        logger.debug('Conv without bias')
        has_bias = False
        W = layers[node.input[1]]
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    is_constant = is_numpy(input_0) or isinstance(input_0, EagerTensor)
    is_W_constant = is_numpy(W) or isinstance(W, EagerTensor)
    n_groups = params['group'] if 'group' in params else 1
    dilations = params.get('dilations', [1])
    dilation = dilations[0]
    dilation_has_unsupported = any(v > 1 for v in dilations)
    pads = params['pads'] if 'pads' in params else [0, 0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1, 1]
    auto_pad = params.get('auto_pad',"".encode()).decode()
    if "SAME" in auto_pad:
        input_size = np.array(input_0.shape[2:]) #Assuming NCHW
        if None in input_size:
            raise Exception("Conv Layers currently does not currently support auto_pad with dynamic input shape")
        else:
            output_size = tf.math.ceil(input_size/np.array(strides))
            kernel_size = np.array(W.shape[2:])
            pads = np.maximum(0, (output_size - 1) * np.array(strides) + dilation * (kernel_size - 1) + 1 - input_size).astype(int)
        pad_before = np.floor(pads/2).astype(int)
        pad_after = pads-pad_before
        if "LOWER" in auto_pad:
            #SAME LOWER means you pad more before
            pad_after, pad_before = pad_before, pad_after
        pads = np.column_stack((pad_before, pad_after)).ravel()
    if len(W.shape) == 5:  # 3D conv
        logger.debug('3D convolution')
        if pads[0] > 0 or pads[1] > 0 or pads[2] > 0:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = f"{params['cleaned_name']}_" + 'conv_pad'
            padding_layer = keras.layers.ZeroPadding3D(
                padding=(pads[0], pads[1], pads[2]),
                name=padding_name
            )
            layers[padding_name] = input_0 = padding_layer(input_0)
        out_channels, channels_per_group, dimension, height, width = W.shape
        W = W.transpose(2, 3, 4, 1, 0)

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]
        conv_args = {"filters": out_channels,
                     "kernel_size": (dimension, height, width),
                     "strides": (strides[0], strides[1], strides[2]),
                     "padding": 'valid',
                     "use_bias": has_bias,
                     "activation": None,
                     "dilation_rate": dilation,
                     "name": f"{params['cleaned_name']}_" + 'conv',
                     "groups": n_groups}
        partial_conv = partial(keras.layers.Conv3D, **conv_args)
        layers[node_name] = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2]*n_groups, params, weights=weights)

    elif len(W.shape) == 4:  # 2D conv
        logger.debug('2D convolution')

        padding = None
        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            padding = (pads[0], pads[1])
        elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
            padding = ((pads[0], pads[2]), (pads[1], pads[3]))

        if padding:
            logger.debug('Paddings exist, add ZeroPadding layer')
            padding_name = f"{params['cleaned_name']}_" + 'conv_pad_1'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=padding,
                name=padding_name,
                data_format='channels_first'
            )
            layers[padding_name] = input_0 = padding_layer(input_0)

        W = W.transpose(2, 3, 1, 0) if is_W_constant else tf.transpose(W, [2, 3, 1, 0])
        height, width, channels_per_group, out_channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]
        if is_W_constant:
            conv_args = {"filters": out_channels,
                         "kernel_size": (height, width),
                         "strides": (strides[0], strides[1]),
                         "padding": 'valid',
                             "use_bias": has_bias,
                         "activation": None,
                         "dilation_rate": dilation,
                         "name": f"{params['cleaned_name']}_" + 'conv',
                         "groups": n_groups}

            partial_conv = partial(keras.layers.Conv2D, **conv_args)
            layers[node_name] = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2]*n_groups, params, weights=weights)
        else:
            input_0_nhwc = tf_transpose(input_0, [0, 2, 3, 1],
                                        tf_name=f"{params['cleaned_name']}_" + 'conv_transpose_nhwc')

            # Perform the convolution in NHWC format
            conv_nhwc = tf.nn.conv2d(input_0_nhwc, weights[0], strides=(strides[0], strides[1]),
                                     dilations=dilation,
                                     padding='VALID', data_format='NHWC',
                                     name=f"{params['cleaned_name']}_" + 'conv')

            # Permute the result back to NCHW format
            layers[node_name] = tf_transpose(conv_nhwc, [0, 3, 1, 2],
                                             tf_name=f"{params['cleaned_name']}_" + 'conv_transpose_2_nchw')
    else:
        # 1D conv
        W = W.transpose(2, 1, 0)
        width, channels, n_filters = W.shape
        print(width, channels, n_filters, has_bias)

        weights = [W]
        conv_args = {"filters": n_filters,
                     "kernel_size": (width),
                     "strides": (strides[0]),
                     "use_bias": False,
                     "activation": None,
                     "dilation_rate": dilation,
                     "name": f"{params['cleaned_name']}_" + 'conv',
                     "groups": n_groups}

        padding = None
        if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
            padding = (pads[0], pads[1])

        if padding:
            # find the dimension to pad and use the exact padding values
            input_shape = np.asarray(input_0.shape)
            partitioned_dim = np.argwhere(input_shape == channels * n_groups)[0][0]
            padding_dim = 2 if partitioned_dim == 1 else 1
            tf_padding = np.zeros((2, len(input_shape))).astype(int)
            tf_padding[:, padding_dim] = [padding[0], padding[1]]
            input_0 = tf_pad(input_0, tf.constant(list(tf_padding.transpose())),
                             tf_name=f"{params['cleaned_name']}_conv_pad_0")
        else:
            conv_args['padding'] = 'valid'
        partial_conv = partial(keras.layers.Conv1D, **conv_args)
        res = permute_wrap_conv_if_constant(partial_conv, input_0, is_constant, weights[0].shape[-2]*n_groups, params, weights=weights)
        if has_bias:
            res_shape = np.asarray(res.shape)
            bias_dim = np.argwhere(res_shape == bias.shape)[0][0]
            expanded_dims = [dim for dim in range(len(res_shape)) if dim != bias_dim]
            res = res + np.expand_dims(bias, expanded_dims)

        layers[node_name] = res


def convert_convtranspose(node, params, layers,
                          lambda_func, node_name, keras_name):
    """
    Convert transposed convolution layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.convtranpose')

    if len(node.input) == 3:
        logger.debug('ConvTranspose with bias')
        # Has bias
        has_bias = True
        W = layers[node.input[1]]
        bias = layers[node.input[2]]

    elif len(node.input) == 2:
        logger.debug('ConvTranspose without bias')
        has_bias = False
        W = layers[node.input[1]]
        bias = None

    else:
        raise NotImplementedError('Not implemented')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    is_W_constant = is_numpy(W) or isinstance(W, EagerTensor)
    n_groups = params['group'] if 'group' in params else 1
    dilations = params.get('dilations', [1])
    dilation = dilations[0]
    dilation_has_unsupported = any(v > 1 for v in dilations)
    pads = params['pads'] if 'pads' in params else [0, 0]
    strides = params['strides'] if 'strides' in params else [1, 1]

    if len(W.shape) == 5:  # 3D conv
        W = W.transpose(2, 3, 4, 1, 0)
        height, width, depth, n_filters, channels = W.shape
        strides_3d = params.get('strides', [1, 1, 1])
        pads_3d = params.get('pads', [0, 0, 0])

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if n_groups > 1:
            if dilation_has_unsupported:
                raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')
            if 'output_padding' in params and any(v > 0 for v in params['output_padding']):
                raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')
            layers[node_name] = grouped_conv_transpose(
                input_0,
                kernel=W,
                bias=bias if has_bias else None,
                strides=strides_3d,
                dilations=params.get('dilations', [1, 1, 1]),
                pads=pads_3d,
                output_padding=params.get('output_padding', [0, 0, 0]),
                groups=n_groups,
                data_format="channels_first",
                trainable_weights=False,
                name=f"{params['cleaned_name']}_convtranspose_grouped"
            )
            return

        if dilation_has_unsupported:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')

        conv = keras.layers.Conv3DTranspose(
            filters=n_filters,
            kernel_size=(height, width, depth),
            strides=strides_3d,
            padding='valid',
            output_padding=0,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation,
            name=f"{params['cleaned_name']}_convtranspose"
        )

        if 'output_shape' in params and 'pads' not in params:
            logger.debug('!!!!! Paddings will be calculated automatically !!!!!')
            pads_3d = [strides_3d[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][0],
                       strides_3d[1] * (int(input_0.shape[3]) - 1) + 0 + (width - 1) * dilation - params['output_shape'][1],
                       strides_3d[2] * (int(input_0.shape[4]) - 1) + 0 + (depth - 1) * dilation - params['output_shape'][2]]

        layers[node_name] = input_0 = conv(input_0)
        conv.set_weights(weights)

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        # input_0.set_shape(input_0.shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if any(pads_3d):
            logger.debug('Add cropping layer for output padding')
            if len(pads_3d) == 3:
                cropping = tuple(pads_3d)
            else:
                assert len(pads_3d) == 6
                assert (pads_3d[3] == pads_3d[0] and pads_3d[4] == pads_3d[1] and pads_3d[5] == pads_3d[2])
                cropping = ((pads_3d[0], pads_3d[3]),
                            (pads_3d[1], pads_3d[4]),
                            (pads_3d[2], pads_3d[5]))

            crop = keras.layers.Cropping3D(
                cropping,
                name=f"{params['cleaned_name']}_convtranspose" + '_crop'
            )
            layers[node_name] = crop(input_0)

    elif len(W.shape) == 4:  # 2D conv
        W = W.transpose(2, 3, 1, 0) if is_W_constant else tf.transpose(W, [2, 3, 1, 0])
        height, width, n_filters, channels = W.shape

        if has_bias:
            weights = [W, bias]
        else:
            weights = [W]

        if n_groups > 1:
            if dilation_has_unsupported:
                raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')
            if 'output_padding' in params and any(v > 0 for v in params['output_padding']):
                raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')
            layers[node_name] = grouped_conv_transpose(
                input_0,
                kernel=W,
                bias=bias if has_bias else None,
                strides=strides,
                dilations=params.get('dilations', [1, 1]),
                pads=pads,
                output_padding=params.get('output_padding', [0, 0]),
                groups=n_groups,
                data_format="channels_first",
                trainable_weights=False,
                name=f"{params['cleaned_name']}_convtranspose_grouped"
            )
            return

        if dilation_has_unsupported:
            raise AttributeError('Cannot convert ConvTranspose2d with dilation_rate != 1')
        if is_W_constant:
            conv = keras.layers.Conv2DTranspose(
                filters=n_filters,
                kernel_size=(height, width),
                strides=strides,
                padding='valid',
                output_padding=0,
                use_bias=has_bias,
                activation=None,
                dilation_rate=dilation,
                name=f"{params['cleaned_name']}_convtranspose"
            )

            if 'output_shape' in params and 'pads' not in params:
                logger.debug('!!!!! Paddings will be calculated automatically !!!!!')
                pads = [
                    strides[0] * (int(input_0.shape[2]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][0],
                    strides[1] * (int(input_0.shape[3]) - 1) + 0 + (height - 1) * dilation - params['output_shape'][1]]

            layers[node_name] = input_0 = conv(input_0)
            conv.set_weights(weights)
        else:
            input_0_nhwc = tf.transpose(input_0, [0, 2, 3, 1])

            output_shape = infer_output_shape(input_shape=tf.shape(input_0_nhwc), filter_shape=tf.shape(W),
                                              strides=strides,
                                              padding='VALID')
            conv_transpose_nhwc = tf.nn.conv2d_transpose(input_0_nhwc, weights[0], output_shape=output_shape,
                                                         strides=(strides[0], strides[1]), dilations=dilation, padding='VALID',
                                                         data_format='NHWC',
                                                         name=f"{params['cleaned_name']}_convtranspose_nhwc")

            # Permute the result back to NCHW format
            layers[node_name] = tf_transpose(conv_transpose_nhwc, [0, 3, 1, 2],
                                             tf_name=f"{params['cleaned_name']}_convtranspose")

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        # input_0.set_shape(input_0.shape)

        if 'output_padding' in params and (params['output_padding'][0] > 0 or params['output_padding'][1] > 0):
            raise AttributeError('Cannot convert ConvTranspose2d with output_padding != 0')

        if pads[0] > 0:
            logger.debug('Add cropping layer for output padding')
            assert (len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=f"{params['cleaned_name']}_convtranspose" + '_crop_1'
            )
            layers[node_name] = crop(input_0)
    else:
        raise AttributeError('Layer is not supported for now')


def infer_output_shape(input_shape, filter_shape, strides, padding):
    input_size_h, input_size_w = input_shape[1:3]
    filter_size_h, filter_size_w = filter_shape[0:2]

    if padding == 'SAME':
        pad_h = max((input_size_h - 1) * strides[0] + filter_size_h - input_size_h, 0) // 2
        pad_w = max((input_size_w - 1) * strides[1] + filter_size_w - input_size_w, 0) // 2
    elif padding == 'VALID':
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("Padding must be 'SAME' or 'VALID'")

    output_size_h = (input_size_h - 1) * strides[0] + filter_size_h - 2 * pad_h
    output_size_w = (input_size_w - 1) * strides[1] + filter_size_w - 2 * pad_w

    return [input_shape[0], output_size_h,
            output_size_w, filter_shape[2]]  # [batch_size, output_channels, output_height, output_width]
