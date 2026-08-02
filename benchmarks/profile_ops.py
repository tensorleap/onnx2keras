"""Per-op-type cost attribution for a converted model, via the TensorFlow profiler.

Profiles the graph-mode forward pass and parses the emitted xplane protobuf
directly. TF 2.12 writes only ``*.xplane.pb`` (no ``trace.json.gz``) and leaves
the ``tf_op`` XStat unpopulated, so op types are taken from the event-name
suffix, e.g. ``model/permute_50/transpose:Transpose``.

Durations are summed per op type across worker threads, so on CPU they are
thread-time rather than wall time and are meaningful as relative shares only.
The executor scope events (ExecutorState::Process and friends) enclose the real
ops and are excluded so they do not double-count.

    poetry run python benchmarks/profile_ops.py model.onnx --logdir /tmp/prof
"""
import argparse
import collections
import glob
import os

import tensorflow as tf
from tensorflow.core.profiler.protobuf import xplane_pb2

from benchmarks.common import bench, convert, describe_devices, keras_args, make_feed, ort_session

SCOPE_EVENTS = ('ExecutorState', 'EagerExecute', 'EagerLocalExecute', 'EagerKernelExecute',
                'KernelAndDeviceFunc', 'TFE_Py_ExecuteCancelable', 'TFE_DeleteTensorHandle',
                'FunctionRun', 'ExecutorDoneCallback', 'EagerCopyToDevice', '<lambda>')


def is_kernel_event(name, strict):
    """Real kernel events are named ``scope/op_name:OpType``.

    Runtime scope events either carry no colon (``<lambda>``) or use a double
    colon (``ExecutorState::Process``), and they enclose the kernels they
    dispatch -- counting them double-counts the entire forward pass. GPU device
    planes name kernels without the colon suffix, so the strict rule is relaxed
    for any plane where it matches nothing.
    """
    if not name or name.startswith(SCOPE_EVENTS):
        return False
    if not strict:
        return True
    return ':' in name and '::' not in name


def aggregate(plane, names, strict):
    totals = collections.Counter()
    counts = collections.Counter()
    for line in plane.lines:
        for event in line.events:
            name = names.get(event.metadata_id, '')
            if not is_kernel_event(name, strict):
                continue
            op_type = name.rsplit(':', 1)[1] if ':' in name else name
            totals[op_type] += event.duration_ps
            counts[op_type] += 1
    return totals, counts


def parse_xplane(logdir, iters):
    paths = sorted(glob.glob(os.path.join(logdir, '**', '*.xplane.pb'), recursive=True))
    if not paths:
        raise FileNotFoundError(f'no xplane.pb written under {logdir}')
    space = xplane_pb2.XSpace()
    space.ParseFromString(open(paths[-1], 'rb').read())

    per_plane = {}
    for plane in space.planes:
        names = {e.id: e.name for e in plane.event_metadata.values()}
        totals, counts = aggregate(plane, names, strict=True)
        if not totals:
            totals, counts = aggregate(plane, names, strict=False)
        if totals:
            per_plane[plane.name] = (totals, counts)
    return per_plane


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model')
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--spatial', type=int, default=640)
    parser.add_argument('--sequence', type=int, default=128)
    parser.add_argument('--dim', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args()
    dims = {k: int(v) for k, v in (d.split('=', 1) for d in args.dim)}

    print(describe_devices())
    sess = ort_session(args.model)
    feed = make_feed(sess, dims, args.spatial, args.sequence)
    t_onnx = bench(lambda f: sess.run(None, f), feed, args.iters, 3, synchronize=False)

    _, keras_model = convert(args.model)
    x = keras_args(feed, keras_model)
    tensors = tf.nest.flatten(x)
    specs = [tf.TensorSpec(t.shape, t.dtype) for t in tensors]
    graph_fn = tf.function(lambda t: keras_model(t)).get_concrete_function(
        specs[0] if len(specs) == 1 else specs)
    t_graph = bench(graph_fn, x, args.iters, 5)

    tf.profiler.experimental.start(args.logdir)
    for _ in range(args.iters):
        graph_fn(x)
    tf.profiler.experimental.stop()

    print(f'\nonnxruntime {t_onnx:.1f} ms | keras graph {t_graph:.1f} ms '
          f'| {t_graph / t_onnx:.2f}x')

    for plane_name, (totals, counts) in parse_xplane(args.logdir, args.iters).items():
        total = sum(totals.values())
        print(f'\n=== {plane_name} === (sum of leaf op time: '
              f'{total / 1e9 / args.iters:.1f} ms/iter, wall was {t_graph:.1f} ms)')
        print(f'{"op":26s} {"n/iter":>8s} {"ms/iter":>9s} {"share":>8s}')
        for op_type, duration in totals.most_common(args.top):
            print(f'{op_type[:26]:26s} {counts[op_type] / args.iters:8.1f} '
                  f'{duration / 1e9 / args.iters:9.2f} {duration / total * 100:7.1f}%')


if __name__ == '__main__':
    main()
