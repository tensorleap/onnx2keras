import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_where():
    cond_info = helper.make_tensor_value_info('condition', TensorProto.BOOL, [2, 3, 4])
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])
    out_info = helper.make_tensor_value_info('out', TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node('Where', inputs=['condition', 'x', 'y'], outputs=['out'])
    model = make_single_op_model('Where', [node], [cond_info, x_info, y_info], [out_info],
                                 opset_version=16)
    condition = (np.random.randn(2, 3, 4) > 0).astype(bool)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'condition': condition, 'x': x, 'y': y}, ['condition', 'x', 'y'])
