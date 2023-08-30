import logging

import numpy as np
import tensorflow as tf

from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM
from onnx2kerastl.customonnxlayer.onnxgru import OnnxGRU
from onnx2kerastl.customonnxlayer.onnxRNN import OnnxRNN
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
    activation_alpha = params.get('activation_alpha')
    activation_beta = params.get('activation_beta')
    activations = params.get('activations')
    clip = params.get('clip')
    batch_first = params.get('layout', 0) == 1
    direction = params.get('direction', b'forward').decode()
    linear_before_reset = params.get('linear_before_reset', 0)
    hidden_size = params.get('hidden_size')
    if activation_alpha is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the activation_alpha param in GRU Node")
    if activation_beta is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the activation_beta param in GRU Node")
    if activations is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the activations param in GRU Node")
    if clip is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the clip param in GRU Node")
    x = layers[node.input[0]]
    w = layers[node.input[1]]
    r = layers[node.input[2]]
    # Since we can't access Optional inputs names we have to use some heuristic to classify them
    b = layers.get(node.input[3])
    if b is None:
        b = np.zeros((w.shape[0], 2*w.shape[1]))
    sequence_len = layers.get(node.input[4])
    if sequence_len is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the Sequence Length param in GRU Node")
    initial_h = layers.get(node.input[5])
   # b, sequence_lens,  initial_h = classify_tensors([first_tensor, second_tensor, third_tensor])
    #linear_before_reset
    if not batch_first:
        x_n = tf.transpose(x, [1, 0, 2])
    tf.keras.backend.set_image_data_format("channels_last")

    if direction == "bidirectional":
        weights = [w.swapaxes(1, 2)[0, ...], r.swapaxes(1, 2)[0, ...], b[0, ...].reshape(2, -1),
                   w.swapaxes(1, 2)[1, ...], r.swapaxes(1, 2)[1, ...], b[1, ...].reshape(2, -1)]
        gru_layer = OnnxGRU(hidden_size, return_sequences=True, return_states=True, bidirectional=True)
        concat_res = gru_layer(x_n, [initial_h[0], initial_h[1]])
    elif direction == "reverse":
        raise AttributeError("Does not currently support reverse direction on GRU layers")
    else:
        weights = [w.swapaxes(1, 2)[0, ...], r.swapaxes(1, 2)[0, ...], b.reshape(2, -1)]
        gru_layer = OnnxGRU(hidden_size, return_sequences=True, return_states=True, bidirectional=False)
        concat_res = gru_layer(x_n, initial_h[0])
    gru_layer.gru_layer.set_weights(weights)
    layers[node.output[0]] = concat_res[:-1, :, :, :]
    layers[node.output[1]] = concat_res[-1, :, :, :]
    tf.keras.backend.set_image_data_format("channels_first")


def convert_rnn(node, params, layers, lambda_func, node_name, keras_name):
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
    activation_alpha = params.get('activation_alpha')
    activation_beta = params.get('activation_beta')
    clip = params.get('clip')
    batch_first = params.get('layout', 0) == 1
    direction = params.get('direction', b'forward').decode()
    activations = params.get('activations', [b'tanh', b'tanh'] if direction == 'bidirectional' else [b'tanh'])
    activations = [act.decode().lower() for act in activations]
    hidden_size = params.get('hidden_size')
    if activation_alpha is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the activation_alpha param in RNN Node")
    if activation_beta is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the activation_beta param in RNN Node")
    if len(activations) > 1:
        if activations[0] != activations[1]:
            raise AttributeError("Onnx2kerras : We do not support different activation in fwd bg pass in RNN")
    activations = activations[0]
    if clip is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the clip param in RNN Node")
    x = layers[node.input[0]]
    w = layers[node.input[1]]
    r = layers[node.input[2]]
    # Since we can't access Optional inputs names we have to use some heuristic to classify them
    b = layers.get(node.input[3])
    if b is None:
        b = np.zeros((w.shape[0], 2*w.shape[1]))
    sequence_len = layers.get(node.input[4])
    if sequence_len is not None:
        raise AttributeError("Onnx2kerras : We do not currently support the Sequence Length param in RNN Node")
    initial_h = layers.get(node.input[5])
    if not batch_first:
        x_n = tf.transpose(x, [1, 0, 2])
    tf.keras.backend.set_image_data_format("channels_last")
    if direction == "bidirectional":
        weights = [w.swapaxes(1, 2)[0, ...], r.swapaxes(1, 2)[0, ...], b[0, ...].reshape(2, -1).sum(axis=0),
                   w.swapaxes(1, 2)[1, ...], r.swapaxes(1, 2)[1, ...], b[1, ...].reshape(2, -1).sum(axis=0)]
        states_vector = [initial_h[0], initial_h[1]]
        gru_layer = OnnxRNN(hidden_size, return_sequences=True, return_states=True, bidirectional=True,
                            activation=activations)
    elif direction == "reverse":
        raise AttributeError("Does not currently support reverse direction on RNN layers")
    else:
        weights = [w.swapaxes(1, 2)[0, ...], r.swapaxes(1, 2)[0, ...], b.reshape(2, -1).sum(axis=0)]
        states_vector = initial_h[0]
        gru_layer = OnnxRNN(hidden_size, return_sequences=True, return_states=True, bidirectional=False,
                            activation=activations)
    concat_res = gru_layer(x_n, states_vector)
    gru_layer.gru_layer.set_weights(weights)
    layers[node.output[0]] = concat_res[:-1, :, :, :]
    layers[node.output[1]] = concat_res[-1, :, :, :]
    tf.keras.backend.set_image_data_format("channels_first")
