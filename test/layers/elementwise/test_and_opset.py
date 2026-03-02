import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_and():
    a_info = helper.make_tensor_value_info('A', TensorProto.BOOL, [2, 3, 4])
    b_info = helper.make_tensor_value_info('B', TensorProto.BOOL, [2, 3, 4])
    y_info = helper.make_tensor_value_info('Y', TensorProto.BOOL, [2, 3, 4])
    node = helper.make_node('And', inputs=['A', 'B'], outputs=['Y'])
    model = make_single_op_model('And', [node], [a_info, b_info], [y_info], opset_version=7)
    a = (np.random.randn(2, 3, 4) > 0).astype(bool)
    b = (np.random.randn(2, 3, 4) > 0).astype(bool)
    run_op_test(model, {'A': a, 'B': b}, ['A', 'B'])
