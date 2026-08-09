import itertools

import numpy as np
import tensorflow as tf
from keras.layers import Layer


def _make_taps(kernel_size):
    return list(itertools.product(*[range(k) for k in kernel_size]))


class _SparseConv3DGeometry:
    """Shared geometry for the sparse-conv pair.

    The active-site bookkeeping ("rulebook") depends only on the input
    coordinates, never on feature values, so the coords layer and the features
    layer can each recompute it independently from the same inputs.
    """

    def _init_geometry(self, kernel_size, stride, padding, dilation, in_shape, subm):
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.stride = tuple(int(s) for s in stride)
        self.padding = tuple(int(p) for p in padding)
        self.dilation = tuple(int(d) for d in dilation)
        self.in_shape = tuple(int(s) for s in in_shape)
        self.subm = bool(subm)
        self._taps = _make_taps(self.kernel_size)

    @property
    def out_shape(self):
        if self.subm:
            return self.in_shape
        return tuple(
            (self.in_shape[d] + 2 * self.padding[d] - self.dilation[d] * (self.kernel_size[d] - 1) - 1)
            // self.stride[d]
            + 1
            for d in range(3)
        )

    @staticmethod
    def _flat_index(coords, shape):
        z, y, x = coords[..., 0], coords[..., 1], coords[..., 2]
        return z * (shape[1] * shape[2]) + y * shape[2] + x

    def _compute_out_coords(self, in_coords):
        """Active output sites, ordered by flattened coordinate."""
        if self.subm:
            return in_coords
        out_shape = tf.constant(self.out_shape, dtype=tf.int64)
        stride_t = tf.constant(self.stride, dtype=tf.int64)
        padding_t = tf.constant(self.padding, dtype=tf.int64)
        dilation_t = tf.constant(self.dilation, dtype=tf.int64)
        taps_t = tf.constant(self._taps, dtype=tf.int64)

        candidates = []
        for k in range(len(self._taps)):
            tap = taps_t[k]
            numer = in_coords + padding_t - tap * dilation_t
            divisible = tf.reduce_all(numer % stride_t == 0, axis=-1)
            cand = numer // stride_t
            inb = tf.reduce_all((cand >= 0) & (cand < out_shape), axis=-1)
            candidates.append(tf.boolean_mask(cand, divisible & inb))
        flat = self._flat_index(tf.concat(candidates, axis=0), out_shape)
        uniq, _ = tf.unique(flat)
        uniq = tf.sort(uniq)
        oz = uniq // (out_shape[1] * out_shape[2])
        rem = uniq % (out_shape[1] * out_shape[2])
        return tf.stack([oz, rem // out_shape[2], rem % out_shape[2]], axis=-1)

    def _geometry_config(self):
        return {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "in_shape": self.in_shape,
            "subm": self.subm,
        }


class TLSparseConv3DCoords(Layer, _SparseConv3DGeometry):
    """Active output coordinates of a 3D sparse convolution.

    Split out from the features layer so that each layer has a SINGLE output:
    keras_data_format_converter walks a model's tensors assuming one output per
    layer and crashes ('tuple' object has no attribute '_keras_history') on
    multi-output layers.
    """

    def __init__(self, kernel_size, stride, padding, dilation, in_shape, subm=False, **kwargs):
        super().__init__(**kwargs)
        self._init_geometry(kernel_size, stride, padding, dilation, in_shape, subm)

    def call(self, inputs, *args, **kwargs):
        return self._compute_out_coords(tf.cast(inputs, tf.int64))

    def get_config(self):
        config = super().get_config()
        config.update(self._geometry_config())
        return config


class TLSparseConv3DFeatures(Layer, _SparseConv3DGeometry):
    """Feature half of a 3D sparse convolution (spconv-style
    SparseConvolution / SubMConv3d), on an explicit (coords, features) pair.

    Verified equivalent to the real compiled spconv CUDA op -- exact coordinate
    match and float32-tolerance feature values -- for submanifold and regular
    (strided) configs, including asymmetric kernel/stride/padding.

    Takes the output coordinates as an input (produced by TLSparseConv3DCoords)
    so that it has a single output.
    """

    def __init__(self, kernel_size, stride, padding, dilation, in_shape,
                 in_channels, out_channels, subm=False, weight=None, bias=None, **kwargs):
        super().__init__(**kwargs)
        self._init_geometry(kernel_size, stride, padding, dilation, in_shape, subm)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self._init_weight = None if weight is None else np.asarray(weight, dtype=np.float32)
        self._init_bias = None if bias is None else np.asarray(bias, dtype=np.float32)

    def build(self, input_shape):
        w_init = ((lambda shape, dtype: tf.constant(self._init_weight, dtype=dtype))
                  if self._init_weight is not None else "glorot_uniform")
        b_init = ((lambda shape, dtype: tf.constant(self._init_bias, dtype=dtype))
                  if self._init_bias is not None else "zeros")
        self.weight = self.add_weight(
            name="weight", shape=(*self.kernel_size, self.in_channels, self.out_channels),
            dtype=tf.float32, trainable=True, initializer=w_init)
        self.bias = self.add_weight(
            name="bias", shape=(self.out_channels,), dtype=tf.float32,
            trainable=True, initializer=b_init)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        in_coords, in_feats, out_coords = inputs
        in_coords = tf.cast(in_coords, tf.int64)
        out_coords = tf.cast(out_coords, tf.int64)
        in_feats = tf.cast(in_feats, tf.float32)

        in_shape = tf.constant(self.in_shape, dtype=tf.int64)
        stride_t = tf.constant(self.stride, dtype=tf.int64)
        padding_t = tf.constant(self.padding, dtype=tf.int64)
        dilation_t = tf.constant(self.dilation, dtype=tf.int64)
        taps_t = tf.constant(self._taps, dtype=tf.int64)

        in_flat = self._flat_index(in_coords, in_shape)
        sort_order = tf.argsort(in_flat)
        sorted_in_flat = tf.gather(in_flat, sort_order)
        n_in = tf.shape(in_coords)[0]

        def lookup_in(query):
            in_bounds = tf.reduce_all((query >= 0) & (query < in_shape), axis=-1)
            q_flat = self._flat_index(query, in_shape)
            pos = tf.searchsorted(sorted_in_flat, q_flat, side="left")
            pos_c = tf.clip_by_value(pos, 0, n_in - 1)
            found = in_bounds & (tf.gather(sorted_in_flat, pos_c) == q_flat) & (pos < n_in)
            return found, tf.gather(sort_order, pos_c)

        acc = tf.zeros((tf.shape(out_coords)[0], self.out_channels), dtype=tf.float32)
        for k in range(len(self._taps)):
            query = out_coords * stride_t - padding_t + taps_t[k] * dilation_t
            found, idx = lookup_in(query)
            contribution = tf.matmul(tf.gather(in_feats, idx), self.weight[self._taps[k]])
            acc += contribution * tf.cast(found, tf.float32)[:, None]
        return acc + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(self._geometry_config())
        config.update({"in_channels": self.in_channels, "out_channels": self.out_channels})
        return config
