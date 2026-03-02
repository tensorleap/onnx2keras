import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_gather_elements():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 3])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 3])
    indices_init = numpy_helper.from_array(
        np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int64), name='indices'
    )
    node = helper.make_node('GatherElements', inputs=['x', 'indices'], outputs=['y'], axis=1)
    model = make_single_op_model('GatherElements', [node], [x_info], [y_info], opset_version=13,
                                 initializers=[indices_init])
    x = np.random.randn(3, 3).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
