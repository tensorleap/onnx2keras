import numpy as np
import tensorflow as tf
from keras.layers import Layer


class TLScatterToDense(Layer):
    """Scatter sparse (coords, features) into a dense grid.

    Exists instead of a plain ONNX ScatterND because ScatterND needs its
    zero-filled target tensor as an explicit graph input. For real BEV grids
    that constant is large (e.g. 1x180x180x2x128 float32 = 33MB, 8.3M values),
    and Keras serializes such captured constants into the layer config as
    nested Python lists -> JSON, which blows memory up by orders of magnitude
    on save. Here the target is allocated inside call() from `dense_shape`,
    so the config carries a handful of ints instead.

    `reduction` selects the semantics:
      - "update": last write wins (spconv's SparseConvTensor.dense())
      - "add":    duplicate indices accumulate (LSS-style bev_pool)

    Args:
        dense_shape: full output shape including the batch axis, e.g.
            [1, 180, 180, 2, 128] -- indices index its leading dims and the
            trailing dims carry the feature vector.
        reduction: "update" or "add"
    """

    def __init__(self, dense_shape, reduction="update", **kwargs):
        super().__init__(**kwargs)
        self.dense_shape = tuple(int(s) for s in dense_shape)
        if reduction not in ("update", "add"):
            raise ValueError(f"unsupported reduction: {reduction}")
        self.reduction = reduction

    def call(self, inputs, *args, **kwargs):
        indices, updates = inputs
        indices = tf.cast(indices, tf.int32)
        updates = tf.cast(updates, tf.float32)

        shape = tf.constant(self.dense_shape, dtype=tf.int32)
        if self.reduction == "add":
            return tf.scatter_nd(indices, updates, shape)

        zeros = tf.zeros(self.dense_shape, dtype=updates.dtype)
        return tf.tensor_scatter_nd_update(zeros, indices, updates)

    def compute_output_shape(self, input_shape):
        return self.dense_shape

    def get_config(self):
        config = super().get_config()
        config.update({"dense_shape": self.dense_shape, "reduction": self.reduction})
        return config
