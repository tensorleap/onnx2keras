import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_gather():
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, 4])
    y_info = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 4])
    indices_init = numpy_helper.from_array(np.array([1, 3], dtype=np.int64), name='indices')
    node = helper.make_node('Gather', inputs=['x', 'indices'], outputs=['y'], axis=0)
    model = make_single_op_model('Gather', [node], [x_info], [y_info], opset_version=13,
                                 initializers=[indices_init])
    x = np.random.randn(5, 4).astype(np.float32)
    run_op_test(model, {'x': x}, ['x'])
