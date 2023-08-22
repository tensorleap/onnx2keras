import logging

import keras
import tensorflow as tf

from .customonnxlayer.onnxerf import OnnxErf
from .customonnxlayer.onnxhardsigmoid import OnnxHardSigmoid
from .utils import ensure_tf_type


def convert_relu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    relu = keras.layers.Activation('relu', name=keras_name)
    layers[node_name] = relu(input_0)


def convert_elu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    alpha = params.get('alpha', keras.layers.ELU.__init__.__defaults__[0])
    elu = keras.layers.ELU(alpha=alpha, name=keras_name)
    layers[node_name] = elu(input_0)


def convert_lrelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert LeakyReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    alpha = params.get('alpha', keras.layers.LeakyReLU.__init__.__defaults__[0])
    leakyrelu = keras.layers.LeakyReLU(alpha=alpha, name=keras_name)
    layers[node_name] = leakyrelu(input_0)


def convert_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    sigmoid = keras.layers.Activation('sigmoid', name=keras_name)
    layers[node_name] = sigmoid(input_0)


def convert_tanh(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Tanh activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    tanh = keras.layers.Activation('tanh', name=keras_name)
    layers[node_name] = tanh(input_0)


def convert_selu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    selu = keras.layers.Activation('selu', name=keras_name)
    layers[node_name] = selu(input_0)


def convert_soft_plus(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.keras.activations.softplus(input_0)


def convert_soft_sign(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.keras.activations.softsign(input_0)


def convert_mish(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = input_0 * tf.math.tanh(tf.math.softplus(input_0))


def convert_hard_swish(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    alpha = 1 / 6
    beta = 0.5
    hard_sigmoid = max(0, min(1, alpha * input_0 + beta))
    hard_swish = input_0 * hard_sigmoid
    layers[node_name] = hard_swish


def convert_gelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert SELU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    layers[node_name] = tf.keras.activations.gelu(input_0)


def convert_softmax(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert softmax activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    axis = params.get('axis', keras.layers.Softmax.__init__.__defaults__[0])
    softmax_layer = keras.layers.Softmax(axis=axis, name=keras_name)
    layers[node_name] = softmax_layer(input_0)
    layers[node_name].set_shape(layers[node_name].shape)


def convert_prelu(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert PReLU activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras.prelu')

    if len(node.input) != 2:
        assert AttributeError('Activation layer PReLU should have 2 inputs.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
    W = layers[node.input[1]]

    if params['change_ordering']:
        logger.warning('PRelu + change ordering needs to be fixed after TF graph is built.')
        logger.warning('It\'s experimental.')

    shared_axes = [2, 3]

    # for case when W.shape (n,). When activation is used for single dimension vector.
    shared_axes = shared_axes if len(W.shape) > 1 else None

    prelu = keras.layers.PReLU(weights=[W], shared_axes=shared_axes, name=keras_name)
    layers[node_name] = prelu(input_0)


def convert_hard_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Hard Sigmoid activation layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    alpha = params.get("alpha", 0.2)
    beta = params.get("beta", 0.5)
    onnx_hard_sigmoid = OnnxHardSigmoid(alpha=alpha, beta=beta, name=keras_name)
    layers[node_name] = onnx_hard_sigmoid(input_0)


def convert_erf(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert ERF math operation
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for an activation layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)

    onnx_erf = OnnxErf(name=keras_name)
    layers[node_name] = onnx_erf(input_0)
