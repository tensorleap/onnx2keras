import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _run(model, feeds, input_names):
    converted = onnx_to_keras(model, input_names=input_names,
                              name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    out = np.array(converted.converted_model([feeds[name] for name in input_names]))
    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(None, feeds)[0]
    return out, ref


@pytest.mark.parametrize('per_channel', [True, False])
def test_fused_groupnormalization(per_channel):
    channels, groups = 32, 4
    affine_len = channels if per_channel else groups
    rng = np.random.default_rng(0)
    scale = numpy_helper.from_array(rng.standard_normal(affine_len).astype(np.float32), 'scale')
    bias = numpy_helper.from_array(rng.standard_normal(affine_len).astype(np.float32), 'bias')
    node = helper.make_node('GroupNormalization', ['x', 'scale', 'bias'], ['y'],
                            num_groups=groups, epsilon=1e-5)
    graph = helper.make_graph(
        [node], 'gn',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', channels, 8])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', channels, 8])],
        [scale, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])

    x = rng.standard_normal((3, channels, 8)).astype(np.float32)
    if per_channel:
        # the installed onnxruntime enforces opset-18 per-group affine params and cannot
        # serve as reference for the opset-21 per-channel layout — verify against numpy
        converted = onnx_to_keras(model, input_names=['x'],
                                  name_policy='attach_weights_name',
                                  allow_partial_compilation=False)
        out = np.array(converted.converted_model([x]))
        grouped = x.reshape(3, groups, -1)
        normalized = (grouped - grouped.mean(axis=2, keepdims=True)) / np.sqrt(
            grouped.var(axis=2, keepdims=True) + 1e-5)
        scale_np = numpy_helper.to_array(scale).reshape(1, channels, 1)
        bias_np = numpy_helper.to_array(bias).reshape(1, channels, 1)
        ref = normalized.reshape(3, channels, 8) * scale_np + bias_np
    else:
        out, ref = _run(model, {'x': x}, ['x'])
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-4)


def test_fused_groupnormalization_dynamic_spatial():
    channels, groups = 32, 4
    rng = np.random.default_rng(0)
    scale = numpy_helper.from_array(rng.standard_normal(groups).astype(np.float32), 'scale')
    bias = numpy_helper.from_array(rng.standard_normal(groups).astype(np.float32), 'bias')
    node = helper.make_node('GroupNormalization', ['x', 'scale', 'bias'], ['y'],
                            num_groups=groups, epsilon=1e-5)
    graph = helper.make_graph(
        [node], 'gn_dyn',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', channels, 'height', 'width'])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', channels, 'height', 'width'])],
        [scale, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])

    x = rng.standard_normal((3, channels, 5, 7)).astype(np.float32)
    out, ref = _run(model, {'x': x}, ['x'])
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-4)


def test_convtranspose_1d():
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.standard_normal((8, 6, 4)).astype(np.float32) * 0.1, 'W')
    bias = numpy_helper.from_array(rng.standard_normal(6).astype(np.float32), 'B')
    node = helper.make_node('ConvTranspose', ['x', 'W', 'B'], ['y'],
                            kernel_shape=[4], strides=[2], pads=[1, 1])
    graph = helper.make_graph(
        [node], 'ct1d',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 8, 16])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 6, 32])],
        [weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])

    x = rng.standard_normal((2, 8, 16)).astype(np.float32)
    out, ref = _run(model, {'x': x}, ['x'])
    assert out.shape == ref.shape == (2, 6, 32)
    assert np.allclose(out, ref, atol=1e-4)
