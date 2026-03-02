import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_concat():
    a_info = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4])
    b_info = helper.make_tensor_value_info('b', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 6, 4])
    node = helper.make_node('Concat', inputs=['a', 'b'], outputs=['y'], axis=1)
    model = make_single_op_model('Concat', [node], [a_info, b_info], [y_info], opset_version=13)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'a': a, 'b': b}, ['a', 'b'])
