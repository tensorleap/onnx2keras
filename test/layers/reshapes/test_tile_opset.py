import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_tile():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 3, 8])
    repeats_init = numpy_helper.from_array(np.array([2, 1, 2], dtype=np.int64), name='repeats')
    node = helper.make_node('Tile', inputs=['x', 'repeats'], outputs=['y'])
    model = make_single_op_model('Tile', [node], [x_info], [y_info], opset_version=13,
                                 initializers=[repeats_init])
    x = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
