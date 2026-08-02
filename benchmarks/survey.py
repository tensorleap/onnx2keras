"""Survey conversion overhead across ONNX models.

Reports, per model, the Keras layer histogram, the Permute-to-conv ratio, the
post-Grappler graph op counts, and onnxruntime / eager / graph / predict timings.

    poetry run python benchmarks/survey.py test/models/yolo_v7/yolov7-tiny.onnx
    poetry run python benchmarks/survey.py --out results.jsonl test/**/*.onnx
"""
import argparse
import collections
import json
import sys

import numpy as np
import tensorflow as tf

from benchmarks.common import (bench, convert, describe_devices, keras_args, make_feed,
                               ort_session)

COMPUTE_PREFIXES = ('Conv', 'Dense', 'Separable', 'Depthwise')
LAMBDA_TYPES = ('TFOpLambda', 'SlicingOpLambda', 'Lambda')


def survey(path, args):
    record = {'model': path}

    sess = ort_session(path, args.providers)
    feed = make_feed(sess, args.dims, args.spatial, args.sequence)
    record['input_shapes'] = {k: list(v.shape) for k, v in feed.items()}
    record['t_onnx'] = bench(lambda f: sess.run(None, f), feed, args.iters, args.warmup,
                             synchronize=False)

    onnx_model, keras_model = convert(path)
    record['onnx_nodes'] = len(onnx_model.graph.node)

    hist = collections.Counter(type(l).__name__ for l in keras_model.layers)
    record['keras_layers'] = len(keras_model.layers)
    record['hist'] = dict(hist.most_common())
    record['n_permute'] = hist.get('Permute', 0)
    record['n_lambda'] = sum(hist.get(t, 0) for t in LAMBDA_TYPES)
    record['n_compute'] = sum(v for k, v in hist.items() if k.startswith(COMPUTE_PREFIXES))

    x = keras_args(feed, keras_model)
    record['t_keras_eager'] = bench(keras_model, x, args.iters, args.warmup)

    tensors = tf.nest.flatten(x)
    specs = [tf.TensorSpec(t.shape, t.dtype) for t in tensors]
    graph_fn = tf.function(lambda t: keras_model(t)).get_concrete_function(
        specs[0] if len(specs) == 1 else specs)
    record['t_keras_graph'] = bench(graph_fn, x, args.iters, args.warmup)

    if len(tensors) == 1:
        numpy_input = list(feed.values())[0]
        record['t_keras_predict'] = bench(
            lambda a: keras_model.predict(a, verbose=0), numpy_input,
            max(args.iters // 2, 2), 1, synchronize=False)

    graph_ops = collections.Counter(o.type for o in graph_fn.graph.get_operations())
    record['graph_transpose'] = graph_ops.get('Transpose', 0)
    record['graph_ops'] = sum(graph_ops.values())
    record['graph_hist'] = dict(graph_ops.most_common(12))

    if record['n_compute']:
        record['permute_per_conv'] = round(record['n_permute'] / record['n_compute'], 2)
    record['slowdown_graph'] = round(record['t_keras_graph'] / record['t_onnx'], 2)
    record['slowdown_eager'] = round(record['t_keras_eager'] / record['t_onnx'], 2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+')
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--spatial', type=int, default=640,
                        help='size for symbolic spatial dims (default 640)')
    parser.add_argument('--sequence', type=int, default=128,
                        help='size for symbolic sequence dims (default 128)')
    parser.add_argument('--dim', action='append', default=[], metavar='NAME=VALUE',
                        help='override a symbolic dim by name, repeatable')
    parser.add_argument('--providers', nargs='+', default=None,
                        help='onnxruntime providers (default: CUDA if available, else CPU)')
    parser.add_argument('--out', default=None, help='append JSONL records here')
    parser.add_argument('--keep-going', action='store_true',
                        help='record and continue past a model that fails to convert')
    args = parser.parse_args()
    args.dims = dict(d.split('=', 1) for d in args.dim)
    args.dims = {k: int(v) for k, v in args.dims.items()}

    print(json.dumps(describe_devices()), file=sys.stderr)

    records = []
    for path in args.models:
        if args.keep_going:
            try:
                record = survey(path, args)
            except Exception as exc:
                record = {'model': path, 'error': f'{type(exc).__name__}: {exc}'}
        else:
            record = survey(path, args)
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.out:
            with open(args.out, 'a') as f:
                f.write(json.dumps(record) + '\n')

    print(f'\n{"model":26s} {"nodes":>6s} {"layers":>7s} {"perm":>5s} {"conv":>5s} '
          f'{"p/c":>5s} {"lam":>5s} {"onnx":>9s} {"graph":>9s} {"eager":>9s} {"x":>6s}',
          file=sys.stderr)
    for r in records:
        if 'error' in r:
            print(f'{r["model"][-26:]:26s} {r["error"][:80]}', file=sys.stderr)
            continue
        print(f'{r["model"][-26:]:26s} {r["onnx_nodes"]:6d} {r["keras_layers"]:7d} '
              f'{r["n_permute"]:5d} {r["n_compute"]:5d} {r.get("permute_per_conv", 0):5.1f} '
              f'{r["n_lambda"]:5d} {r["t_onnx"]:8.1f}m {r["t_keras_graph"]:8.1f}m '
              f'{r["t_keras_eager"]:8.1f}m {r["slowdown_graph"]:5.2f}x', file=sys.stderr)


if __name__ == '__main__':
    main()
