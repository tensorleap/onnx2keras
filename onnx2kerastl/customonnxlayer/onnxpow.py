import tensorflow as tf
from keras.layers import Layer


class OnnxPow(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if len(inputs) != 2:
            raise ValueError(
                'A `Pow` layer should be called on exactly 2 inputs. '
                f'Received: inputs={inputs}')
        x = tf.math.pow(inputs[0], inputs[1])
        return x
