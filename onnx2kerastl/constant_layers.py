import numpy as np
import tensorflow as tf
from .utils import is_numpy
from .tfops_funcs import tf_cast, tf_one_hot
import keras

def convert_constant(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert Constant layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    layers[node_name] = params['value']


def convert_constant_of_shape(node, params, layers, lambda_func, node_name, keras_name):
    value = params.get('value')
    if value is None:
        # Per ONNX spec, default value is 0.0 (float32) when not specified
        value = np.array([0.0], dtype=np.float32)

    input_0 = layers[node.input[0]]

    if not is_numpy(input_0) and not isinstance(input_0, list) and isinstance(input_0, keras.KerasTensor):
        # Boolean case
        if value.dtype == np.bool_:
            layers[node_name] = tf.fill(layers[node.input[0]], tf.constant(value.item(), dtype=tf.bool))
        else:
            # Non-boolean case
            layers[node_name] = tf.ones(layers[node.input[0]], dtype=tf.as_dtype(value.dtype)) * value
    else:
        # Handle numpy inputs or non-Keras tensors
        if value.dtype == np.bool_:
            layers[node_name] = np.full(layers[node.input[0]], value.item(), dtype=bool)
        else:
            layers[node_name] = np.ones(layers[node.input[0]], dtype=value.dtype) * value



class _OneHotLayer(keras.layers.Layer):
    """Custom layer for one_hot that properly reports output shape."""
    def __init__(self, depth, on_value, off_value, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis

    def call(self, x):
        return tf.one_hot(x, depth=self.depth, on_value=self.on_value,
                          off_value=self.off_value, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.axis == -1:
            return tuple(input_shape + [self.depth])
        else:
            return tuple(input_shape[:self.axis] + [self.depth] + input_shape[self.axis:])

    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'on_value': self.on_value,
            'off_value': self.off_value,
            'axis': self.axis,
        })
        return config


def convert_one_hot(node, params, layers, lambda_func, node_name, keras_name):
    axis = params.get('axis', -1)
    depth = int(layers[node.input[1]])
    off_value = float(layers[node.input[2]][0])
    on_value = float(layers[node.input[2]][1])
    indices = tf_cast(layers[node.input[0]], tf.int32,
                      tf_name=f"{params['cleaned_name']}_onehot_cast")

    one_hot_layer = _OneHotLayer(depth=depth, on_value=on_value, off_value=off_value,
                                  axis=axis, name=f"{params['cleaned_name']}_onehot")
    layers[node_name] = one_hot_layer(indices)

