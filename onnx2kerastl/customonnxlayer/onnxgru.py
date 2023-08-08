from keras.layers import Layer
import tensorflow as tf


class OnnxGRU(Layer):
    """

    Args:
        units: int
        return_sequences: bool
        return_states: bool
        bidirectional: bool
        **kwargs:
    """

    def __init__(self, units: int, return_sequences: bool, return_states: bool, bidirectional: bool,  **kwargs):
        super().__init__(**kwargs)
        self.gru_layer = tf.keras.layers.GRU(units, return_sequences=return_sequences,
                                               return_state=return_states)
        if bidirectional:
            self.gru_layer = tf.keras.layers.Bidirectional(self.gru_layer)
        self.return_state = return_states
        self.return_sequences = return_sequences
        self.units = units
        self.bidirectional = bidirectional

    def call(self, inputs, initial_h_state=None, **kwargs):
        res = self.gru_layer(inputs, initial_h_state, **kwargs)
        if self.bidirectional:
            gru_res, forward_state, backward_state = res
            res_unsqueeze_direction = tf.reshape(gru_res, (*gru_res.shape[:2], 2, -1))
            res = tf.transpose(res_unsqueeze_direction, (1, 2, 0, 3))
            out_states = tf.concat([tf.expand_dims(forward_state, 0), tf.expand_dims(backward_state, 0)], axis=0)
        else:
            gru_res, states = res
            out_states = tf.expand_dims(states, 0)
            gru_permuted = tf.transpose(gru_res, (1, 0, 2))
            res = tf.expand_dims(gru_permuted, axis=1)
        concat_res = tf.concat([res, tf.expand_dims(out_states, 0)], axis=0)
        return concat_res

    def build(self, input_shape):
        self.gru_layer.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "return_sequences": self.return_sequences,
            "return_states": self.return_state,
            "units": self.units,
            "bidirectional": self.bidirectional

        })
        return config
