import numpy as np
import tensorflow as tf
from typing import Callable
from keras import backend as K
import keras
import logging
logger = logging.getLogger('onnx2keras.tfops_funcs')

layer_names_counter = {}


def _is_keras_tensor(x):
    return isinstance(x, keras.KerasTensor)


def _has_keras_tensor(*args, **kwargs):
    """Check if any argument (nested in lists/tuples) is a KerasTensor."""
    for a in args:
        if _is_keras_tensor(a):
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if _is_keras_tensor(item):
                    return True
    for v in kwargs.values():
        if _is_keras_tensor(v):
            return True
        if isinstance(v, (list, tuple)):
            for item in v:
                if _is_keras_tensor(item):
                    return True
    return False


def _set_layer_name(result, tf_name):
    """Set the layer name on the operation that produced this tensor (Keras 2/3 compatible)."""
    if isinstance(result, (tf.Tensor, np.ndarray)):
        return
    try:
        # Keras 3: use _keras_history
        op = result._keras_history.operation
        if op is not None:
            op._name = tf_name
    except (AttributeError, TypeError):
        try:
            # Keras 2 fallback: use .node.layer
            result.node.layer._name = tf_name
        except (AttributeError, TypeError):
            pass


class _TFOpLayer(keras.layers.Layer):
    """Wraps a TF function as a Keras Layer. Unlike Lambda, call() receives real
    tensors (not KerasTensors) so any TF op can be used inside."""

    def __init__(self, func, frozen_args, frozen_kwargs, kt_indices, **kwargs):
        super().__init__(**kwargs)
        self._func = func
        self._frozen_args = frozen_args
        self._frozen_kwargs = frozen_kwargs
        self._kt_indices = kt_indices

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        restored_args = list(self._frozen_args)
        restored_kwargs = dict(self._frozen_kwargs)
        input_idx = 0
        for idx_info in self._kt_indices:
            if idx_info[0] == 'arg':
                restored_args[idx_info[1]] = inputs[input_idx]
                input_idx += 1
            elif idx_info[0] == 'list_arg':
                lst = list(restored_args[idx_info[1]])
                lst[idx_info[2]] = inputs[input_idx]
                restored_args[idx_info[1]] = lst
                input_idx += 1
            elif idx_info[0] == 'kwarg':
                restored_kwargs[idx_info[1]] = inputs[input_idx]
                input_idx += 1
        return self._func(*restored_args, **restored_kwargs)

    def compute_output_shape(self, input_shape):
        # Infer the output shape by tracing the function with concrete TensorSpecs.
        try:
            if isinstance(input_shape, list):
                specs = [tf.TensorSpec(shape=s, dtype=tf.float32) for s in input_shape]
            else:
                specs = tf.TensorSpec(shape=input_shape, dtype=tf.float32)

            @tf.function
            def _trace(x):
                if isinstance(x, (list, tuple)):
                    return self.call(list(x))
                return self.call(x)

            concrete = _trace.get_concrete_function(specs)
            out_shape = concrete.output_shapes
            # out_shape may be a TensorShape or a tuple/list of TensorShapes
            if isinstance(out_shape, (list, tuple)):
                return out_shape[0]
            return out_shape
        except Exception:
            # Fallback: return the first input shape unchanged
            if isinstance(input_shape, list):
                return input_shape[0]
            return input_shape

    def get_config(self):
        config = super().get_config()
        config['func_name'] = getattr(self._func, '__name__', str(self._func))
        return config


def named_tfop(func: Callable):
    def wrapped_function(*args, tf_name=None, **kwargs):
        if tf_name is None or tf_name == "" or not isinstance(tf_name, str):
            tf_name = func.__name__ if hasattr(func, '__name__') else "unnamed_op"
        # Keras 3 forbids '/' in layer names; sanitize
        tf_name = tf_name.replace("/", "_")

        if not _has_keras_tensor(*args, **kwargs):
            # No KerasTensors — call directly (numpy/eager mode)
            return func(*args, **kwargs)

        # Try calling the TF op directly (works for most ops in Keras 3 with TF backend)
        try:
            result = func(*args, **kwargs)
        except (ValueError, TypeError) as e:
            if "KerasTensor" in str(e):
                # Op doesn't support KerasTensors — wrap in a Layer subclass
                # (not Lambda, because Lambda traces with KerasTensors too)
                kt_inputs = []
                kt_indices = []
                for i, a in enumerate(args):
                    if _is_keras_tensor(a):
                        kt_indices.append(('arg', i))
                        kt_inputs.append(a)
                    elif isinstance(a, (list, tuple)):
                        for j, item in enumerate(a):
                            if _is_keras_tensor(item):
                                kt_indices.append(('list_arg', i, j))
                                kt_inputs.append(item)
                for key, val in kwargs.items():
                    if _is_keras_tensor(val):
                        kt_indices.append(('kwarg', key))
                        kt_inputs.append(val)

                frozen_args = list(args)
                frozen_kwargs = dict(kwargs)

                layer = _TFOpLayer(func, frozen_args, frozen_kwargs, kt_indices, name=tf_name)
                if len(kt_inputs) == 1:
                    result = layer(kt_inputs[0])
                else:
                    result = layer(kt_inputs)
                return result
            else:
                raise

        # Deduplicate names
        if tf_name not in layer_names_counter:
            layer_names_counter[tf_name] = 0
        else:
            layer_names_counter[tf_name] += 1
            tf_name = f"{tf_name}_{layer_names_counter[tf_name]}"

        _set_layer_name(result, tf_name)
        return result
    return wrapped_function


tf_cast = named_tfop(tf.cast)
tf_shape = named_tfop(tf.shape)
tf_math_abs = named_tfop(tf.math.abs)
tf_reshape = named_tfop(tf.reshape)
tf_stack = named_tfop(tf.stack)
tf_add = named_tfop(tf.add)
tf_image_resize = named_tfop(tf.image.resize)
tf_multiply = named_tfop(tf.multiply)
tf_clip_by_value = named_tfop(tf.clip_by_value)
tf_math_negative = named_tfop(tf.math.negative)
tf_tensor_scatter_nd_update = named_tfop(tf.tensor_scatter_nd_update)
K_mean = named_tfop(tf.math.reduce_mean)
tf_math_reduce_prod = named_tfop(tf.math.reduce_prod)
tf_math_reduce_min = named_tfop(tf.math.reduce_min)
tf_math_pow = named_tfop(tf.math.pow)
tf_math_sqrt = named_tfop(tf.math.sqrt)
tf_strided_slice = named_tfop(tf.strided_slice)
tf_squeeze = named_tfop(tf.squeeze)
tf_argmax = named_tfop(tf.argmax)
tf_expand_dims = named_tfop(tf.expand_dims)
tf_maximum = named_tfop(tf.maximum)
tf_minimum = named_tfop(tf.minimum)
tf_repeat = named_tfop(tf.repeat)
tf_matmul = named_tfop(tf.matmul)
tf_concat = named_tfop(tf.concat)
tf_transpose = named_tfop(tf.transpose)
tf_math_reduce_variance = named_tfop(tf.math.reduce_variance)
tf_math_reduce_mean = named_tfop(tf.math.reduce_mean)
tf_sqrt = named_tfop(tf.sqrt)
tf_where = named_tfop(tf.where)
tf_gather = named_tfop(tf.gather)
tf_range = named_tfop(tf.range)
tf_abs = named_tfop(tf.abs)
tf_reduce_sum = named_tfop(tf.reduce_sum)
tf_pad = named_tfop(tf.pad)
tf_math_erf = named_tfop(tf.math.erf)
tf_math_reciprocal = named_tfop(tf.math.reciprocal)
tf_logical_not = named_tfop(tf.logical_not)
tf_equal = named_tfop(tf.equal)
tf_tile = named_tfop(tf.tile)
tf_math_minimum = named_tfop(tf.math.minimum)
tf_math_maximum = named_tfop(tf.math.maximum)
tf_math_sign = named_tfop(tf.math.sign)
tf_math_sin = named_tfop(tf.math.sin)
tf_math_cosh = named_tfop(tf.math.cosh)
tf_math_ceil = named_tfop(tf.math.ceil)
tf_math_acosh = named_tfop(tf.math.acosh)
tf_math_acos = named_tfop(tf.math.acos)
tf_math_asinh = named_tfop(tf.math.asinh)
tf_math_asin = named_tfop(tf.math.asin)
tf_math_atanh = named_tfop(tf.math.atanh)
tf_math_tan = named_tfop(tf.math.tan)
tf_math_atan = named_tfop(tf.math.atan)
tf_math_sinh = named_tfop(tf.math.sinh)
tf_math_less = named_tfop(tf.math.less)
tf_math_less_equal = named_tfop(tf.math.less_equal)
tf_bitwise_invert = named_tfop(tf.bitwise.invert)
tf_bitwise_bitwise_and = named_tfop(tf.bitwise.bitwise_and)
tf_bitwise_bitwise_or = named_tfop(tf.bitwise.bitwise_or)
tf_bitwise_bitwise_xor = named_tfop(tf.bitwise.bitwise_xor)
tf_cos = named_tfop(tf.cos)
tf_math_greater = named_tfop(tf.math.greater)
tf_math_greater_equal = named_tfop(tf.math.greater_equal)
tf_logical_and = named_tfop(tf.logical_and)
tf_math_logical_xor = named_tfop(tf.math.logical_xor)
tf_math_logical_or = named_tfop(tf.math.logical_or)
tf_argmin = named_tfop(tf.argmin)
tf_one_hot = named_tfop(tf.one_hot)
tf_round = named_tfop(tf.round)
tf_math_cumsum = named_tfop(tf.math.cumsum)
tf_math_is_inf = named_tfop(tf.math.is_inf)
tf_math_is_nan = named_tfop(tf.math.is_nan)
tf_size = named_tfop(tf.size)
tf_linalg_det = named_tfop(tf.linalg.det)
tf_not_equal = named_tfop(tf.not_equal)
tf_gather_nd = named_tfop(tf.gather_nd)
tf_math_softplus = named_tfop(tf.math.softplus)
tf_math_tanh = named_tfop(tf.math.tanh)
tf_signal_irfft = named_tfop(tf.signal.irfft)
tf_signal_ifft = named_tfop(tf.signal.ifft)
tf_signal_rfft = named_tfop(tf.signal.rfft)
tf_signal_fft = named_tfop(tf.signal.fft)
tf_sign = named_tfop(tf.sign)
tf_abs = named_tfop(tf.abs)
tf_math_mod = named_tfop(tf.math.mod)
tf_bitwise_left_shift = named_tfop(tf.bitwise.left_shift)
tf_bitwise_right_shift = named_tfop(tf.bitwise.right_shift)
tf_rank = named_tfop(tf.rank)
tf_fill = named_tfop(tf.fill)
tf_image_non_max_suppression = named_tfop(tf.image.non_max_suppression)
tf_ones_like = named_tfop(tf.ones_like)
tf_image_crop_and_resize = named_tfop(tf.image.crop_and_resize)
tf_ones = named_tfop(tf.ones)
tf_math_floor = named_tfop(tf.math.floor)
tf_zeros_like = named_tfop(tf.zeros_like)
tf_tensor_scatter_nd_update = named_tfop(tf.tensor_scatter_nd_update)
tf_nn_avg_pool = named_tfop(tf.nn.avg_pool)
tf_nn_max_pool = named_tfop(tf.nn.max_pool)
tf_linalg_matmul = named_tfop(tf.linalg.matmul)
tf_divide = named_tfop(tf.divide)
tf_subtract = named_tfop(tf.subtract)
