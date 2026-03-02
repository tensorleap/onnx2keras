import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_shape():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 3, 4])
    shape_out_info = helper.make_tensor_value_info('shape_out', TensorProto.INT64, [3])
    node = helper.make_node('Shape', inputs=['x'], outputs=['shape_out'])
    model = make_single_op_model('Shape', [node], [x_info], [shape_out_info], opset_version=13)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
