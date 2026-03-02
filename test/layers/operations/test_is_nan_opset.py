import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_is_nan():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.BOOL, [1, 4])
    node = helper.make_node('IsNaN', inputs=['x'], outputs=['y'])
    model = make_single_op_model('IsNaN', [node], [x_info], [y_info], opset_version=9)
    x = np.array([[np.nan, 1.0, np.nan, 2.0]], dtype=np.float32)
    run_op_test(model, {'x': x}, ['x'])
