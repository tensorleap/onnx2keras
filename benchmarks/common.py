import time

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl import onnx_to_keras

ORT_DTYPES = {
    'tensor(float)': np.float32,
    'tensor(double)': np.float64,
    'tensor(float16)': np.float16,
    'tensor(int64)': np.int64,
    'tensor(int32)': np.int32,
    'tensor(bool)': np.bool_,
}

HEIGHT_NAMES = {'h', 'height', 'img_h', 'image_height', 'ny'}
WIDTH_NAMES = {'w', 'width', 'img_w', 'image_width', 'nx'}
SEQUENCE_NAMES = {'seq', 'seq_len', 'sequence', 'sequence_length', 'tokens', 'num_tokens'}


def resolve_dim(dim_name, axis, rank, overrides, spatial, sequence):
    """Resolve one symbolic ONNX dimension to a concrete size.

    Axis 0 is always treated as batch and resolved to 1 regardless of its name.
    This is deliberate: names such as ``batch_size`` and ``unk__251`` are not
    recognisable from a fixed table, and defaulting them to a spatial size
    silently produces batch-224 benchmarks.
    """
    key = str(dim_name).lower()
    if key in overrides:
        return overrides[key]
    if axis == 0:
        return 1
    if key in HEIGHT_NAMES or key in WIDTH_NAMES:
        return spatial
    if key in SEQUENCE_NAMES:
        return sequence
    return spatial if rank >= 4 else sequence


def make_feed(sess, overrides=None, spatial=640, sequence=128, seed=0):
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    rng = np.random.default_rng(seed)
    feed = {}
    for inp in sess.get_inputs():
        rank = len(inp.shape)
        shape = [
            d if isinstance(d, int) else resolve_dim(d, axis, rank, overrides, spatial, sequence)
            for axis, d in enumerate(inp.shape)
        ]
        dtype = ORT_DTYPES.get(inp.type, np.float32)
        if np.issubdtype(dtype, np.floating):
            feed[inp.name] = rng.random(shape).astype(dtype)
        else:
            feed[inp.name] = np.ones(shape, dtype=dtype)
    return feed


def ort_session(path, providers=None):
    if providers is None:
        available = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available]
    return ort.InferenceSession(path, providers=providers)


def convert(path):
    """Run the production conversion pipeline (see onnx2kerastl/convert_model.py)."""
    model = onnx.load(path)
    initializers = {n.name for n in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializers]
    keras_model = onnx_to_keras(model, input_names, name_policy='attach_weights_name',
                                allow_partial_compilation=False, verbose=False).converted_model
    return model, convert_channels_first_to_last(keras_model,
                                                 should_transform_inputs_and_outputs=False,
                                                 verbose=False)


def sync(result):
    """Force completion of async device work so timings are not understated on GPU."""
    for tensor in tf.nest.flatten(result):
        if hasattr(tensor, '_numpy'):
            tensor._numpy()
        elif hasattr(tensor, 'numpy'):
            tensor.numpy()


def bench(fn, arg, iters=10, warmup=3, synchronize=True):
    for _ in range(warmup):
        out = fn(arg)
        if synchronize:
            sync(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(arg)
        if synchronize:
            sync(out)
    return (time.perf_counter() - start) / iters * 1000


def keras_args(feed, keras_model):
    """Order the ONNX feed to match the Keras model's input order."""
    tensors = [tf.constant(v) for v in feed.values()]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def describe_devices():
    gpus = tf.config.list_physical_devices('GPU')
    return {
        'tf_version': tf.__version__,
        'ort_version': ort.__version__,
        'tf_gpus': [g.name for g in gpus],
        'ort_providers': ort.get_available_providers(),
    }
