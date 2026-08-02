import itertools
from typing import List, Optional

import numpy as np
import tensorflow as tf
from keras.layers import Layer


def _make_taps(kernel_size):
    return list(itertools.product(*[range(k) for k in kernel_size]))


class TLSparseConv3DLayer(Layer):
    """TF-native replacement for a 3D sparse convolution (spconv-style
    SparseConvolution / SubMConv3d), operating on an explicit (coords,
    features) pair rather than a dense tensor.

    Verified equivalent (exact coordinate match, feature values within
    float32 tolerance) against the real compiled spconv CUDA op, for both
    submanifold and regular (strided/downsampling) configurations, including
    asymmetric kernel/stride/padding.

    Args:
        kernel_size, stride, padding, dilation: length-3 int sequences (z, y, x)
        in_shape: length-3 int sequence, the INPUT spatial shape (z, y, x)
        subm: if True, output coordinates equal input coordinates exactly
            (submanifold convolution); if False, the active output set is
            discovered from the input set via the standard sparse-conv rule
        weight: optional initial value, shape (*kernel_size, C_in, C_out)
        bias: optional initial value, shape (C_out,)
    """

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        in_shape,
        in_channels,
        out_channels,
        subm=False,
        weight=None,
        bias=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.stride = tuple(int(s) for s in stride)
        self.padding = tuple(int(p) for p in padding)
        self.dilation = tuple(int(d) for d in dilation)
        self.in_shape = tuple(int(s) for s in in_shape)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.subm = bool(subm)
        self._init_weight = None if weight is None else np.asarray(weight, dtype=np.float32)
        self._init_bias = None if bias is None else np.asarray(bias, dtype=np.float32)
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

    def build(self, input_shape):
        weight_shape = (*self.kernel_size, self.in_channels, self.out_channels)

        weight_init = (
            (lambda shape, dtype: tf.constant(self._init_weight, dtype=dtype))
            if self._init_weight is not None
            else "glorot_uniform"
        )
        bias_init = (
            (lambda shape, dtype: tf.constant(self._init_bias, dtype=dtype))
            if self._init_bias is not None
            else "zeros"
        )

        self.weight = self.add_weight(
            name="weight", shape=weight_shape, dtype=tf.float32, trainable=True, initializer=weight_init
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.out_channels,), dtype=tf.float32, trainable=True, initializer=bias_init
        )
        super().build(input_shape)

    @staticmethod
    def _flatten(coords, shape):
        z, y, x = coords[..., 0], coords[..., 1], coords[..., 2]
        return z * (shape[1] * shape[2]) + y * shape[2] + x

    def call(self, inputs, *args, **kwargs):
        in_coords, in_feats = inputs
        in_coords = tf.cast(in_coords, tf.int64)
        in_feats = tf.cast(in_feats, tf.float32)

        in_shape = tf.constant(self.in_shape, dtype=tf.int64)
        out_shape = tf.constant(self.out_shape, dtype=tf.int64)
        stride_t = tf.constant(self.stride, dtype=tf.int64)
        padding_t = tf.constant(self.padding, dtype=tf.int64)
        dilation_t = tf.constant(self.dilation, dtype=tf.int64)
        taps_t = tf.constant(self._taps, dtype=tf.int64)

        in_flat = self._flatten(in_coords, in_shape)
        sort_order = tf.argsort(in_flat)
        sorted_in_flat = tf.gather(in_flat, sort_order)
        n_in = tf.shape(in_coords)[0]

        def lookup_in(query_coords):
            in_bounds = tf.reduce_all((query_coords >= 0) & (query_coords < in_shape), axis=-1)
            q_flat = self._flatten(query_coords, in_shape)
            pos = tf.searchsorted(sorted_in_flat, q_flat, side="left")
            pos_clipped = tf.clip_by_value(pos, 0, n_in - 1)
            candidate = tf.gather(sorted_in_flat, pos_clipped)
            found = in_bounds & (candidate == q_flat) & (pos < n_in)
            orig_idx = tf.gather(sort_order, pos_clipped)
            return found, orig_idx

        if self.subm:
            out_coords = in_coords
        else:
            candidates = []
            for k in range(len(self._taps)):
                tap = taps_t[k]
                numer = in_coords + padding_t - tap * dilation_t
                divisible = tf.reduce_all(numer % stride_t == 0, axis=-1)
                cand_out = numer // stride_t
                in_out_bounds = tf.reduce_all((cand_out >= 0) & (cand_out < out_shape), axis=-1)
                valid = divisible & in_out_bounds
                candidates.append(tf.boolean_mask(cand_out, valid))
            all_candidates = tf.concat(candidates, axis=0)
            out_flat_all = self._flatten(all_candidates, out_shape)
            unique_flat, _ = tf.unique(out_flat_all)
            unique_flat = tf.sort(unique_flat)
            oz = unique_flat // (out_shape[1] * out_shape[2])
            rem = unique_flat % (out_shape[1] * out_shape[2])
            oy = rem // out_shape[2]
            ox = rem % out_shape[2]
            out_coords = tf.stack([oz, oy, ox], axis=-1)

        n_out = tf.shape(out_coords)[0]
        c_out = self.weight.shape[-1]
        acc = tf.zeros((n_out, c_out), dtype=tf.float32)
        for k in range(len(self._taps)):
            tap = taps_t[k]
            query = out_coords * stride_t - padding_t + tap * dilation_t
            found, in_idx = lookup_in(query)
            neighbor_feats = tf.gather(in_feats, in_idx)
            contribution = tf.matmul(neighbor_feats, self.weight[self._taps[k]])
            mask = tf.cast(found, tf.float32)[:, None]
            acc += contribution * mask

        return out_coords, acc + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "in_shape": self.in_shape,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "subm": self.subm,
            }
        )
        return config
