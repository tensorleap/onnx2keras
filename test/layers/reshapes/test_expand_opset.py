import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_expand():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 1])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])
    shape_init = numpy_helper.from_array(np.array([2, 3, 4], dtype=np.int64), name='shape')
    node = helper.make_node('Expand', inputs=['x', 'shape'], outputs=['y'])
    model = make_single_op_model('Expand', [node], [x_info], [y_info], opset_version=13,
                                 initializers=[shape_init])
    x = np.random.randn(1, 3, 1).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
