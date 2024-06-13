import logging

import numpy as np
import tensorflow as tf

from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM
from .exceptions import UnsupportedLayer
from .utils import ensure_tf_type


def convert_lstm(node, params, layers, lambda_func, node_name, keras_name):
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

    if node.input[4] != '':
        raise UnsupportedLayer('LSTM with non default sequence_lens')
    if 'direction' in params:
        direction = params['direction']
        if isinstance(direction, bytes):
            direction = direction.decode("utf-8")
        if direction != 'forward':
            raise UnsupportedLayer(f"LSTM with {direction} direction")
    should_return_state = len(node.output) == 3
    input_tensor = tf.transpose(ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name[0]), perm=[1, 0, 2])
    weights_w = layers[node.input[1]][0]
    weights_r = layers[node.input[2]][0]
    weights_b = layers[node.input[3]][0]

    initial_h_state = tf.cast(tf.squeeze(ensure_tf_type(layers[node.input[5]]), axis=0), input_tensor.dtype)
    initial_c_state = tf.cast(tf.squeeze(ensure_tf_type(layers[node.input[6]]), axis=0), input_tensor.dtype)

    tf.keras.backend.set_image_data_format("channels_last")
    hidden_size = params['hidden_size']
    lstm_layer = OnnxLSTM(hidden_size, return_sequences=True, return_lstm_state=should_return_state)
    res = lstm_layer(input_tensor, initial_h_state, initial_c_state)
    # prepare the keras lstm weights from the onnx inputs:
    w1 = np.concatenate([weights_w[0:hidden_size, :], weights_w[2 * hidden_size:3 * hidden_size, :],
                         weights_w[3 * hidden_size:4 * hidden_size, :],
                         weights_w[hidden_size:2 * hidden_size, :]]).transpose()
    w2 = np.concatenate([weights_r[0:hidden_size, :], weights_r[2 * hidden_size:3 * hidden_size, :],
                         weights_r[3 * hidden_size:4 * hidden_size, :],
                         weights_r[hidden_size:2 * hidden_size, :]]).transpose()
    weights_b_part1 = weights_b[:w2.shape[1]]
    weights_b_part2 = weights_b[w2.shape[1]:]
    bias1 = np.concatenate([weights_b_part1[0:hidden_size], weights_b_part1[2 * hidden_size:3 * hidden_size],
                            weights_b_part1[3 * hidden_size:4 * hidden_size],
                            weights_b_part1[hidden_size:2 * hidden_size]]).transpose()
    bias2 = np.concatenate([weights_b_part2[0:hidden_size], weights_b_part2[2 * hidden_size:3 * hidden_size],
                            weights_b_part2[3 * hidden_size:4 * hidden_size],
                            weights_b_part2[hidden_size:2 * hidden_size]]).transpose()
    bias = bias1 + bias2
    res.node.layer.set_weights([w1, w2, bias])
    tf.keras.backend.set_image_data_format("channels_first")
    if should_return_state:
        c_out = res[:, -1, :]
        h_out = res[:, 0, :]

        # the shapes of the hidden and cell should be [num_directions, batch_size, hidden_size]
        # for now we support only direction=forward so num_direction = 1 and we add directions dimension,
        # if we support direction=bidirectional we should handle it well in the lstm layer and probably remove the
        # expand dims here
        c_out = tf.expand_dims(c_out, 0)
        h_out = tf.expand_dims(h_out, 0)

        lstm_tensor = res[:, 1:-1, :]
        layers[node.output[1]] = h_out
        layers[node.output[2]] = c_out
    else:
        lstm_tensor = res
    lstm_tensor_in_onnx_order = tf.transpose(lstm_tensor, perm=[1, 0, 2])
    lstm_tensor_in_onnx_order = tf.expand_dims(lstm_tensor_in_onnx_order, axis=1)
    layers[node_name] = lstm_tensor_in_onnx_order

def convert_gru(node, params, layers, lambda_func, node_name, keras_name):
    logger = logging.getLogger('onnx2keras.convert_gru')
    if len(params["_outputs"]) > 1:
        logger.warning("The GRU return hidden state is currently not supported. Accessing in deeper layers will raise Exception")
    if params.get('activation_alpha') or params.get('activation_beta') or params.get('activations'):
        raise NotImplementedError('Custom Activations in GRU not implemented')
    if params.get('clip'):
        raise NotImplementedError('Clip in GRU not implemented')
    if params.get('direction'):  #After implementation - verify weights reshaping, and h default_size for all directions
        raise NotImplementedError('direction in  GRU not implemented')
    else:
        num_directions = 1
    if params.get('layout'):
        raise NotImplementedError('GRU layout not supported (currently supporting opset 7)')
    else:
        layout = 0
    if node.input[4] != "":
        raise NotImplementedError('GRU sequence_lens is not yet implemented')
    hidden_size = params.get('hidden_size')
    linear_before_reset = bool(params.get('linear_before_reset', 0))
    x = layers[node.input[0]] # [seq_length, batch_size, input_size] iff layout = 0
    w = layers[node.input[1]]
    r = layers[node.input[2]]
    b = layers.get(node.input[3], np.zeros((num_directions, 6*hidden_size), dtype=np.float32))
    h = layers.get(node.input[5], np.zeros((1, x.shape[1], hidden_size), dtype=np.float32))
    if isinstance(h, np.ndarray):
        tensor_h = tf.convert_to_tensor(h)
    else:
        tensor_h = h
    tf.keras.backend.set_image_data_format("channels_last")
    gru_layer = tf.keras.layers.GRU(units=hidden_size,
                        reset_after=linear_before_reset,
                        return_sequences=True)
    if layout == 0:
        batch_first_x = tf.transpose(x, [1, 0, 2])
    res = gru_layer(batch_first_x, initial_state=tf.convert_to_tensor(tensor_h[0]))
    # gru_layer.build(tf.shape(batch_first_x))
    gru_layer.set_weights([w[0].swapaxes(0, 1), r[0].swapaxes(0, 1), b[0].reshape(-1, 3*hidden_size)])
    # res = gru_layer(batch_first_x, initial_state=tf.convert_to_tensor(tensor_h[0]))
    if num_directions == 1:
        reshaped_res = tf.expand_dims(tf.transpose(res, [1, 0, 2]), axis=1)
    else:
        raise NotImplementedError("GRU bidirectional output reshaping is not implemented")
    layers[node_name] = reshaped_res
    tf.keras.backend.set_image_data_format("channels_first")
