import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_cumsum():
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4])
    axis_info = helper.make_tensor_value_info('axis', TensorProto.INT32, [])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node('CumSum', inputs=['data', 'axis'], outputs=['y'],
                            exclusive=0, reverse=0)
    model = make_single_op_model('CumSum', [node], [data_info, axis_info], [y_info],
                                 opset_version=14)
    data = np.random.randn(2, 3, 4).astype(np.float32)
    axis = np.array(1, dtype=np.int32)
    run_op_test(model, {'data': data, 'axis': axis}, ['data', 'axis'])
