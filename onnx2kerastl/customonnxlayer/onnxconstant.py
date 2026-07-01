import numpy as np
import tensorflow as tf
from keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="onnx2kerastl")
class OnnxConstant(Layer):
    """Holds a (large) constant tensor as a non-trainable weight.

    Large constants that are baked directly into the functional graph get
    serialized inline into the model config (Keras' ``_CONSTANT_VALUE``). On
    ``load_model`` TensorFlow rebuilds them by autopacking the nested python
    list element-by-element, which is effectively unbounded for big arrays and
    freezes the load. Storing the value as a layer weight instead keeps it out
    of the config JSON (it goes into the h5 weight datasets) so the model
    round-trips cheaply. ``call`` ignores its input - the input only anchors the
    layer in the functional graph.
    """

    def __init__(self, const_shape=None, const_dtype=None, value=None, **kwargs):
        super().__init__(**kwargs)
        self._value = None if value is None else np.asarray(value)
        if self._value is not None:
            const_shape = list(self._value.shape)
            const_dtype = self._value.dtype.name
        self.const_shape = const_shape
        self.const_dtype = const_dtype

    def build(self, input_shape):
        initializer = tf.constant_initializer(self._value) if self._value is not None else "zeros"
        self.const = self.add_weight(name="const", shape=self.const_shape,
                                     dtype=self.const_dtype, initializer=initializer,
                                     trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.identity(self.const)

    def get_config(self):
        config = super().get_config()
        config.update({"const_shape": self.const_shape, "const_dtype": self.const_dtype})
        return config
