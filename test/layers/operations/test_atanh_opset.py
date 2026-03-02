import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_atanh():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node('Atanh', inputs=['x'], outputs=['y'])
    model = make_single_op_model('Atanh', [node], [x_info], [y_info], opset_version=9)
    x = np.random.uniform(-0.9, 0.9, (2, 3, 4)).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
