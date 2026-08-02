"""Standalone re-timing of individual layers in a converted model.

Recovers each layer's real input shape by running a sub-model up to it, then
times that layer in isolation on a fresh tf.Variable of the same shape. Reports
totals per layer type, and the transpose patterns and bytes moved by Permutes.

Inputs must be real Variables rather than tf.zeros: TensorFlow constant-folds
``tf.transpose(tf.zeros(...))`` at trace time, which makes transposes look free.

    poetry run python benchmarks/layer_costs.py model.onnx
    poetry run python benchmarks/layer_costs.py model.onnx --types Permute Conv2D
"""
import argparse
import collections

import numpy as np
import tensorflow as tf

from benchmarks.common import bench, convert, describe_devices, keras_args, make_feed, ort_session

DEFAULT_TYPES = ('Permute', 'Conv2D', 'Conv3D', 'LeakyReLU', 'Activation', 'ZeroPadding2D',
                 'ZeroPadding3D', 'Concatenate', 'MaxPooling2D', 'BatchNormalization')


def layer_inputs(keras_model, layers, x):
    """Concrete input tensors for each layer, handling multi-input layers."""
    probes = [l.input for l in layers]
    sub = tf.keras.Model(keras_model.input, probes)
    outputs = sub(x)
    if len(probes) == 1:
        outputs = [outputs]
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model')
    parser.add_argument('--types', nargs='+', default=list(DEFAULT_TYPES))
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--spatial', type=int, default=640)
    parser.add_argument('--sequence', type=int, default=128)
    parser.add_argument('--dim', action='append', default=[], metavar='NAME=VALUE')
    args = parser.parse_args()
    dims = {k: int(v) for k, v in (d.split('=', 1) for d in args.dim)}

    print(describe_devices())
    sess = ort_session(args.model)
    feed = make_feed(sess, dims, args.spatial, args.sequence)
    _, keras_model = convert(args.model)
    x = keras_args(feed, keras_model)
    t_full = bench(keras_model, x, 10, 3)
    print(f'full model (eager): {t_full:.1f} ms\n')

    print(f'{"layer type":20s} {"n":>5s} {"ms":>9s} {"share":>8s}')
    for layer_type in args.types:
        layers = [l for l in keras_model.layers if type(l).__name__ == layer_type]
        if not layers:
            continue
        total = 0.0
        for layer, concrete in zip(layers, layer_inputs(keras_model, layers, x)):
            probes = concrete if isinstance(concrete, list) else [concrete]
            variables = [tf.Variable(tf.random.uniform([int(d) for d in p.shape], dtype=p.dtype)
                                     if p.dtype.is_floating else
                                     tf.ones([int(d) for d in p.shape], dtype=p.dtype))
                         for p in probes]
            call_arg = variables if len(variables) > 1 else variables[0]
            fn = tf.function(lambda v, _layer=layer: _layer(v))
            total += bench(fn, call_arg, args.iters, 2)
        print(f'{layer_type:20s} {len(layers):5d} {total:9.2f} {total / t_full * 100:7.1f}%')

    permutes = [l for l in keras_model.layers if type(l).__name__ == 'Permute']
    if not permutes:
        return
    outputs = layer_inputs(keras_model, permutes, x)
    moved = sum(int(np.prod([int(d) for d in o.shape])) * o.dtype.size for o in outputs)
    patterns = collections.Counter(tuple(l.dims) for l in permutes)
    adjacent = sum(1 for l in permutes if type(l.input._keras_history.layer).__name__ == 'Permute')
    print(f'\nPermutes: {len(permutes)}  bytes moved/pass: {moved / 1e6:.1f} MB  '
          f'adjacent pairs: {adjacent}')
    print(f'patterns: {patterns.most_common()}')


if __name__ == '__main__':
    main()
