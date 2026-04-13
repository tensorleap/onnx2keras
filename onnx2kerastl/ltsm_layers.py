import logging

import numpy as np
import tensorflow as tf

from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM
from .exceptions import UnsupportedLayer
from .utils import ensure_tf_type
from .tfops_funcs import tf_cast, tf_squeeze, tf_transpose, tf_expand_dims


def _get_lstm_direction(params):
    direction = params.get('direction', 'forward')
    if isinstance(direction, bytes):
        direction = direction.decode("utf-8")
    return direction


def _prepare_lstm_weights(weights, hidden_size):
    return np.concatenate([
        weights[0:hidden_size, :],
        weights[2 * hidden_size:3 * hidden_size, :],
        weights[3 * hidden_size:4 * hidden_size, :],
        weights[hidden_size:2 * hidden_size, :],
    ]).transpose()


def _prepare_lstm_bias(weights_b, hidden_size):
    weights_b_part1 = weights_b[:4 * hidden_size]
    weights_b_part2 = weights_b[4 * hidden_size:]
    bias1 = np.concatenate([
        weights_b_part1[0:hidden_size],
        weights_b_part1[2 * hidden_size:3 * hidden_size],
        weights_b_part1[3 * hidden_size:4 * hidden_size],
        weights_b_part1[hidden_size:2 * hidden_size],
    ]).transpose()
    bias2 = np.concatenate([
        weights_b_part2[0:hidden_size],
        weights_b_part2[2 * hidden_size:3 * hidden_size],
        weights_b_part2[3 * hidden_size:4 * hidden_size],
        weights_b_part2[hidden_size:2 * hidden_size],
    ]).transpose()
    return bias1 + bias2


def _get_lstm_states(layers, node, input_tensor, params, num_directions):
    if len(node.input) <= 5 or node.input[5] == '':
        initial_h_state = tf.zeros(
            (num_directions, tf.shape(input_tensor)[0], params['hidden_size']),
            dtype=input_tensor.dtype
        )
    else:
        initial_h_state = tf_cast(
            ensure_tf_type(layers[node.input[5]]),
            input_tensor.dtype,
            tf_name=f"{params['cleaned_name']}_lstm_cast_h"
        )

    if len(node.input) <= 6 or node.input[6] == '':
        initial_c_state = tf.zeros(
            (num_directions, tf.shape(input_tensor)[0], params['hidden_size']),
            dtype=input_tensor.dtype
        )
    else:
        initial_c_state = tf_cast(
            ensure_tf_type(layers[node.input[6]]),
            input_tensor.dtype,
            tf_name=f"{params['cleaned_name']}_lstm_cast_c"
        )

    if num_directions == 1:
        initial_h_state = tf_squeeze(
            initial_h_state, axis=0, tf_name=f"{params['cleaned_name']}_lstm_squeeze_h"
        )
        initial_c_state = tf_squeeze(
            initial_c_state, axis=0, tf_name=f"{params['cleaned_name']}_lstm_squeeze_c"
        )

    return initial_h_state, initial_c_state


def _sequence_to_onnx_output(sequence_tensor, num_directions, hidden_size, cleaned_name):
    if num_directions == 1:
        sequence_tensor = tf_expand_dims(
            tf_transpose(sequence_tensor, perm=[1, 0, 2], tf_name=f"{cleaned_name}_lstm_transpose"),
            axis=1,
            tf_name=f"{cleaned_name}_lstm_expand_dims"
        )
        return sequence_tensor

    forward_sequence = sequence_tensor[:, :, :hidden_size]
    backward_sequence = sequence_tensor[:, :, hidden_size:]
    forward_sequence = tf_transpose(
        forward_sequence, perm=[1, 0, 2], tf_name=f"{cleaned_name}_lstm_forward_transpose"
    )
    backward_sequence = tf_transpose(
        backward_sequence, perm=[1, 0, 2], tf_name=f"{cleaned_name}_lstm_backward_transpose"
    )
    return tf.stack([forward_sequence, backward_sequence], axis=1, name=f"{cleaned_name}_lstm_stack")


def _states_to_onnx_output(states, cleaned_name, suffix):
    return tf.stack(states, axis=0, name=f"{cleaned_name}_lstm_{suffix}_state_stack")


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
    direction = _get_lstm_direction(params)
    if direction not in {'forward', 'bidirectional'}:
        raise UnsupportedLayer(f"LSTM with {direction} direction")
    should_return_state = len(node.output) == 3
    num_directions = 2 if direction == 'bidirectional' else 1
    input_tensor = tf_transpose(ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name[0]),
                                perm=[1, 0, 2],
                                tf_name=f"{params['cleaned_name']}_lstm_first_transpose")
    weights_w = layers[node.input[1]]
    weights_r = layers[node.input[2]]
    if len(node.input) > 3 and node.input[3] != '':
        weights_b = layers[node.input[3]]
    else:
        weights_b = np.zeros((num_directions, 8 * params['hidden_size']), dtype=np.float32)

    initial_h_state, initial_c_state = _get_lstm_states(layers, node, input_tensor, params, num_directions)

    tf.keras.backend.set_image_data_format("channels_last")
    hidden_size = params['hidden_size']
    lstm_layer = OnnxLSTM(
        hidden_size,
        return_sequences=True,
        return_lstm_state=should_return_state,
        direction=direction
    )
    res = lstm_layer(input_tensor, initial_h_state, initial_c_state)

    keras_weights = []
    for direction_idx in range(num_directions):
        keras_weights.extend([
            _prepare_lstm_weights(weights_w[direction_idx], hidden_size),
            _prepare_lstm_weights(weights_r[direction_idx], hidden_size),
            _prepare_lstm_bias(weights_b[direction_idx], hidden_size),
        ])
    lstm_layer.set_weights(keras_weights)
    tf.keras.backend.set_image_data_format("channels_first")

    if should_return_state:
        if num_directions == 1:
            h_out = tf_expand_dims(res[:, 0, :], axis=0, tf_name=f"{params['cleaned_name']}_lstm_h_expand_dims")
            c_out = tf_expand_dims(res[:, -1, :], axis=0, tf_name=f"{params['cleaned_name']}_lstm_c_expand_dims")
        else:
            h_concat = res[:, 0, :]
            c_concat = res[:, -1, :]
            h_out = _states_to_onnx_output(
                [h_concat[:, :hidden_size], h_concat[:, hidden_size:]], params['cleaned_name'], "h"
            )
            c_out = _states_to_onnx_output(
                [c_concat[:, :hidden_size], c_concat[:, hidden_size:]], params['cleaned_name'], "c"
            )
        lstm_tensor = res[:, 1:-1, :]
    else:
        lstm_tensor = res

    # Add identical dense contains the lstm tensor for easy fetch of latent space
    input_dim = int(lstm_tensor.shape[2])
    dense = tf.keras.layers.Dense(
        units=input_dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Identity()
    )

    lstm_tensor_dense = dense(lstm_tensor)

    if should_return_state:
        mul_o = lstm_tensor_dense[0, 0, 0] * 0
        c_out = tf.add(c_out, mul_o)
        h_out = tf.add(h_out, mul_o)

        layers[node.output[1]] = h_out
        layers[node.output[2]] = c_out

    lstm_tensor = lstm_tensor_dense

    layers[node_name] = _sequence_to_onnx_output(
        lstm_tensor, num_directions, hidden_size, params['cleaned_name']
    )


def convert_gru(node, params, layers, lambda_func, node_name, keras_name):
    logger = logging.getLogger('onnx2keras.convert_gru')
    if len(params["_outputs"]) > 1:
        logger.warning(
            "The GRU return hidden state is currently not supported. Accessing in deeper layers will raise Exception")
    if params.get('activation_alpha') or params.get('activation_beta') or params.get('activations'):
        raise NotImplementedError('Custom Activations in GRU not implemented')
    if params.get('clip'):
        raise NotImplementedError('Clip in GRU not implemented')
    if params.get(
            'direction'):  # After implementation - verify weights reshaping, and h default_size for all directions
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
    x = layers[node.input[0]]  # [seq_length, batch_size, input_size] iff layout = 0
    w = layers[node.input[1]]
    r = layers[node.input[2]]
    b = layers.get(node.input[3], np.zeros((num_directions, 6 * hidden_size), dtype=np.float32))
    h = layers.get(node.input[5], np.zeros((1, x.shape[1] if x.shape[1] is not None else 1, hidden_size), dtype=np.float32))
    if isinstance(h, np.ndarray):
        tensor_h = tf.convert_to_tensor(h)
    else:
        tensor_h = h
    tf.keras.backend.set_image_data_format("channels_last")
    gru_layer = tf.keras.layers.GRU(units=hidden_size,
                                    reset_after=linear_before_reset,
                                    return_sequences=True,
                                    name=f"{params['cleaned_name']}_gru")
    if layout == 0:
        batch_first_x = tf_transpose(x, [1, 0, 2], tf_name=f"{params['cleaned_name']}_gru_transpose")
    res = gru_layer(batch_first_x, initial_state=tf.convert_to_tensor(tensor_h[0]))
    # gru_layer.build(tf.shape(batch_first_x))
    gru_layer.set_weights([w[0].swapaxes(0, 1), r[0].swapaxes(0, 1), b[0].reshape(-1, 3 * hidden_size)])
    # res = gru_layer(batch_first_x, initial_state=tf.convert_to_tensor(tensor_h[0]))
    if num_directions == 1:
        reshaped_res = tf_expand_dims(tf_transpose(res,
                                                   [1, 0, 2],
                                                   tf_name=f"{params['cleaned_name']}_gru_transpose"),
                                      axis=1,
                                      tf_name=f"{params['cleaned_name']}")
    else:
        raise NotImplementedError("GRU bidirectional output reshaping is not implemented")
    layers[node_name] = reshaped_res
    tf.keras.backend.set_image_data_format("channels_first")
