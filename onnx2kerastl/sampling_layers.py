import numpy as np
import keras
import tensorflow as tf
from keras import backend as K

from .tfops_funcs import tf_cast, tf_range, tf_size
from .customonnxlayer.onnxgridsample import OnnxGridSampleLayer


def convert_range(node, params, layers, lambda_func, node_name, keras_name):
    start_range = layers[node.input[0]]
    limit_range = layers[node.input[1]]
    delta_range = layers[node.input[2]]
    layers[node_name] = tf_range(start_range, limit_range, delta_range, tf_name=f"{params['cleaned_name']}_range")


def convert_gridsample(node, params, layers, lambda_func, node_name, keras_name):
    assert params['mode'].decode('ascii') == 'bilinear'
    assert params['padding_mode'].decode('ascii') == 'zeros'
    align_corners = params.get('align_corners', 0)
    img = layers[node.input[0]]
    sample_grid = layers[node.input[1]]

    grid_is_keras = not isinstance(sample_grid, np.ndarray) and K.is_keras_tensor(sample_grid)

    if grid_is_keras:
        layers[node_name] = OnnxGridSampleLayer(
            align_corners=align_corners,
            name=params['cleaned_name'],
        )([img, sample_grid])
    else:
        layers[node_name] = OnnxGridSampleLayer(
            align_corners=align_corners,
            name=params['cleaned_name'],
            constant_grid=sample_grid,
        )(img)

def random_uniform_like(node, params, layers, lambda_func, node_name, keras_name):
    ret = tf.random.uniform(tf.shape(layers[node.input[0]]))
    layers[node_name] = ret

def convert_unique(node, params, layers, lambda_func, node_name, keras_name):
    to_sort = params.get('sorted', 1) == 1
    axis = params.get('axis')
    if axis is not None:
        raise AttributeError("Onnx2kerras: Unique does not currently support an operation on a non-flattened array")
    lambda_input = layers[node.input[0]]
    rev_idx_length = tf_size(lambda_input, tf_name=f"{params['cleaned_name']}_unique_size_1")

    def target_layer(x):
        input_keras = x
        if axis is None:
            input_final = tf.reshape(input_keras, [-1])
        res, rev_idx, count = tf.unique_with_counts(input_final)
        idx = tf.math.unsorted_segment_min(tf.range(tf.shape(rev_idx)[0]), rev_idx, tf.shape(res)[0])
        if to_sort:
            linspace = tf.range(tf.shape(count)[0])
            argsorted = tf.argsort(res)
            lookup_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(linspace, argsorted),
                                                     default_value=-1)

            rev_idx_sorted = lookup_table.lookup(rev_idx)

            res_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), res, tf.shape(res))
            count_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), count, tf.shape(res))
            idx_sorted = tf.scatter_nd(tf.expand_dims(argsorted, -1), idx, tf.shape(res))
            return tf.concat([tf.cast(rev_idx_sorted, tf.float32), res_sorted, tf.cast(idx_sorted, tf.float32),
                              tf.cast(count_sorted, tf.float32)],
                             axis=0)
        else:
            return tf.concat([tf.cast(rev_idx, tf.float32), res, tf.cast(idx, tf.float32), tf.cast(count, tf.float32)],
                             axis=0)

    lambda_layer = keras.layers.Lambda(target_layer, name=f"{params['cleaned_name']}_unique")
    lambda_res = lambda_layer(lambda_input)
    rev_idx = lambda_res[:rev_idx_length]
    lambda_length = tf_size(lambda_res, tf_name=f"{params['cleaned_name']}_unique_size_2")
    remainder = tf_cast((lambda_length - rev_idx_length) / 3, tf.int32, tf_name=f"{params['cleaned_name']}_unique_cast1")  # not working need to fix
    count = tf_cast(lambda_res[-remainder:], tf.int32, tf_name=f"{params['cleaned_name']}_unique_cast2")
    idx = tf_cast(lambda_res[-2 * remainder:-remainder], tf.int32, tf_name=f"{params['cleaned_name']}_unique_cast3")
    res = tf_cast(lambda_res[-3 * remainder:-2 * remainder], tf.int32, tf_name=f"{params['cleaned_name']}_unique_cast4")
    layers[keras_name[0]] = res
    layers[keras_name[1]] = idx
    layers[keras_name[2]] = rev_idx
    layers[keras_name[3]] = count
