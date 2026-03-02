import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_greater_equal():
    a_info = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 3, 4])
    b_info = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('Y', TensorProto.BOOL, [2, 3, 4])
    node = helper.make_node('GreaterOrEqual', inputs=['A', 'B'], outputs=['Y'])
    model = make_single_op_model('GreaterOrEqual', [node], [a_info, b_info], [y_info],
                                 opset_version=16)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'A': a, 'B': b}, ['A', 'B'])
