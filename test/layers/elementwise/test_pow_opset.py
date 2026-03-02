import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_pow():
    a_info = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3, 4])
    b_info = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node('Pow', inputs=['A', 'B'], outputs=['Y'])
    model = make_single_op_model('Pow', [node], [a_info, b_info], [y_info], opset_version=14)
    a = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    b = np.random.uniform(0.5, 2.0, (2, 3, 4)).astype(np.float32)
    run_op_test(model, {'A': a, 'B': b}, ['A', 'B'])
