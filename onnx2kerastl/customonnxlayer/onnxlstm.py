from keras.layers import Layer
import tensorflow as tf


class OnnxLSTM(Layer):
    """

    Args:
        units: int
        return_sequences: bool
        return_lstm_state: bool
        **kwargs:
    """

    def __init__(self, units: int, return_sequences: bool, return_lstm_state: bool, direction: str = "forward", **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        if direction == "bidirectional":
            self.lstm_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=return_sequences, return_state=return_lstm_state),
                merge_mode="concat"
            )
        else:
            self.lstm_layer = tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                return_state=return_lstm_state
            )
        self.return_lstm_state = return_lstm_state
        self.return_sequences = return_sequences
        self.units = units

    def call(self, inputs, initial_h_state=None, initial_c_state=None, **kwargs):
        if initial_h_state is not None and initial_c_state is not None:
            if self.direction == "bidirectional":
                initial_states = [
                    initial_h_state[0], initial_c_state[0],
                    initial_h_state[1], initial_c_state[1],
                ]
            else:
                initial_states = [initial_h_state, initial_c_state]
        else:
            initial_states = None
        res = self.lstm_layer(inputs, initial_state=initial_states, **kwargs)
        if self.return_lstm_state:
            if self.direction == "bidirectional":
                lstm_tensor, h_forward, c_forward, h_backward, c_backward = res
                h_out = tf.concat([h_forward, h_backward], axis=-1)
                c_out = tf.concat([c_forward, c_backward], axis=-1)
            else:
                lstm_tensor, h_out, c_out = res
            concat_output = tf.concat([tf.expand_dims(h_out, 1), lstm_tensor, tf.expand_dims(c_out, 1)], axis=1)
            return concat_output
        else:
            return res

    def build(self, input_shape):
        self.lstm_layer.build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "return_sequences": self.return_sequences,
            "return_lstm_state": self.return_lstm_state,
            "units": self.units,
            "direction": self.direction
        })
        return config
