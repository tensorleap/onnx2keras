import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_reduce_mean_model_attr(axes, keepdims, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4])
    out_info = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [2, 1, 4])
    node = helper.make_node('ReduceMean', inputs=['data'], outputs=['reduced'],
                            axes=axes, keepdims=keepdims)
    return make_single_op_model('ReduceMean', [node], [data_info], [out_info], opset_version)


def _make_reduce_mean_model_input(axes, keepdims, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4])
    out_info = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [2, 1, 4])
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name='axes')
    node = helper.make_node('ReduceMean', inputs=['data', 'axes'], outputs=['reduced'],
                            keepdims=keepdims)
    return make_single_op_model('ReduceMean', [node], [data_info], [out_info], opset_version,
                                initializers=[axes_init])


@pytest.mark.parametrize('opset_version', [1, 11])
def test_reduce_mean_axes_attr(opset_version):
    data = np.random.randn(2, 3, 4).astype(np.float32)
    model = _make_reduce_mean_model_attr([1], 1, opset_version)
    run_op_test(model, {'data': data}, ['data'])


@pytest.mark.parametrize('opset_version', [18])
def test_reduce_mean_axes_input(opset_version):
    data = np.random.randn(2, 3, 4).astype(np.float32)
    model = _make_reduce_mean_model_input([1], 1, opset_version)
    run_op_test(model, {'data': data}, ['data'])
