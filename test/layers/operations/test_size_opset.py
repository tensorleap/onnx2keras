import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_size():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info('size', TensorProto.INT64, [])
    node = helper.make_node('Size', inputs=['x'], outputs=['size'])
    model = make_single_op_model('Size', [node], [x_info], [y_info], opset_version=1)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
