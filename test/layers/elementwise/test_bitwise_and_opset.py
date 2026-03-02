import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_bitwise_and():
    a_info = helper.make_tensor_value_info('A', TensorProto.INT32, [2, 3, 4])
    b_info = helper.make_tensor_value_info('B', TensorProto.INT32, [2, 3, 4])
    y_info = helper.make_tensor_value_info('Y', TensorProto.INT32, [2, 3, 4])
    node = helper.make_node('BitwiseAnd', inputs=['A', 'B'], outputs=['Y'])
    model = make_single_op_model('BitwiseAnd', [node], [a_info, b_info], [y_info],
                                 opset_version=18)
    a = np.random.randint(0, 100, (2, 3, 4)).astype(np.int32)
    b = np.random.randint(0, 100, (2, 3, 4)).astype(np.int32)
    run_op_test(model, {'A': a, 'B': b}, ['A', 'B'])
