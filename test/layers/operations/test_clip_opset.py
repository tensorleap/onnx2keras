import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_clip_model_attr(min_val, max_val, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('clipped', TensorProto.FLOAT, [3, 4])
    node = helper.make_node('Clip', inputs=['x'], outputs=['clipped'],
                            min=min_val, max=max_val)
    return make_single_op_model('Clip', [node], [x_info], [out_info], opset_version)


def _make_clip_model_input(min_val, max_val, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('clipped', TensorProto.FLOAT, [3, 4])
    min_init = numpy_helper.from_array(np.array(min_val, dtype=np.float32), name='min_val')
    max_init = numpy_helper.from_array(np.array(max_val, dtype=np.float32), name='max_val')
    node = helper.make_node('Clip', inputs=['x', 'min_val', 'max_val'], outputs=['clipped'])
    return make_single_op_model('Clip', [node], [x_info], [out_info], opset_version,
                                initializers=[min_init, max_init])


@pytest.mark.parametrize('opset_version', [1, 6])
def test_clip_attr(opset_version):
    x = np.random.randn(3, 4).astype(np.float32)
    model = _make_clip_model_attr(-0.5, 0.5, opset_version)
    run_op_test(model, {'x': x}, ['x'])


@pytest.mark.parametrize('opset_version', [11, 13])
def test_clip_input(opset_version):
    x = np.random.randn(3, 4).astype(np.float32)
    model = _make_clip_model_input(-0.5, 0.5, opset_version)
    run_op_test(model, {'x': x}, ['x'])
