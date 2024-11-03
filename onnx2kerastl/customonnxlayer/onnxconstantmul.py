from keras.layers import Layer
import tensorflow as tf

class ONNXMultiplyByConstantLayer(Layer):
    def __init__(self, constant_shape, constant_value, **kwargs):
        super(ONNXMultiplyByConstantLayer, self).__init__(**kwargs)
        self.constant_shape = constant_shape
        self.constant_value = constant_value

    def call(self, inputs):
        constant_tensor = tf.fill(self.constant_shape, self.constant_value)
        return inputs * constant_tensor

    def get_config(self):
        config = super(ONNXMultiplyByConstantLayer, self).get_config()
        config.update({
            'constant_shape': self.constant_shape,
            'constant_value': self.constant_value,
        })
        return config