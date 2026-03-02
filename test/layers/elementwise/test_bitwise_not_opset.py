import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_bitwise_not():
    x_info = helper.make_tensor_value_info('x', TensorProto.INT32, [2, 3, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.INT32, [2, 3, 4])
    node = helper.make_node('BitwiseNot', inputs=['x'], outputs=['y'])
    model = make_single_op_model('BitwiseNot', [node], [x_info], [y_info], opset_version=18)
    x = np.random.randint(0, 100, (2, 3, 4)).astype(np.int32)
    run_op_test(model, {'x': x}, ['x'])
