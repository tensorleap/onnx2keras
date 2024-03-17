import tensorflow as tf
from keras.layers import Layer


class OnnxLess(Layer):
    def __init__(self, x, **kwargs):
        super(OnnxLess, self).__init__(**kwargs)
        # Store x as a class attribute
        self.x = x

    def call(self, input, **kwargs):
        # Use the stored x and the input y to perform the less operation
        # Ensure x is cast to the same dtype as input for consistency
        x_casted = tf.cast(self.x, input.dtype)
        return tf.math.less(x_casted, input)

    def get_config(self):
        config = super(OnnxLess, self).get_config()
        # Include the custom layer's initialization arguments
        config.update({
            'x': self.x,
        })
        return config
