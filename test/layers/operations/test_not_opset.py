import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_not():
    x_info = helper.make_tensor_value_info('x', TensorProto.BOOL, [2, 3])
    y_info = helper.make_tensor_value_info('y', TensorProto.BOOL, [2, 3])
    node = helper.make_node('Not', inputs=['x'], outputs=['y'])
    model = make_single_op_model('Not', [node], [x_info], [y_info], opset_version=1)
    x = np.array([[True, False, True], [False, True, False]], dtype=bool)
    run_op_test(model, {'x': x}, ['x'])
