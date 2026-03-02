import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_pad_model_attr(pads, mode, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    out_info = helper.make_tensor_value_info('padded', TensorProto.FLOAT, [3, 4])
    node = helper.make_node('Pad', inputs=['x'], outputs=['padded'],
                            pads=pads, mode=mode)
    return make_single_op_model('Pad', [node], [x_info], [out_info], opset_version)


def _make_pad_model_input(pads, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3])
    out_info = helper.make_tensor_value_info('padded', TensorProto.FLOAT, [3, 4])
    pads_init = numpy_helper.from_array(np.array(pads, dtype=np.int64), name='pads')
    node = helper.make_node('Pad', inputs=['x', 'pads'], outputs=['padded'],
                            mode='constant')
    return make_single_op_model('Pad', [node], [x_info], [out_info], opset_version,
                                initializers=[pads_init])


@pytest.mark.parametrize('opset_version', [2])
def test_pad_attr(opset_version):
    x = np.random.randn(2, 3).astype(np.float32)
    model = _make_pad_model_attr([0, 0, 1, 1], 'constant', opset_version)
    run_op_test(model, {'x': x}, ['x'])


@pytest.mark.parametrize('opset_version', [11, 13])
def test_pad_input(opset_version):
    x = np.random.randn(2, 3).astype(np.float32)
    model = _make_pad_model_input([0, 0, 1, 1], opset_version)
    run_op_test(model, {'x': x}, ['x'])
