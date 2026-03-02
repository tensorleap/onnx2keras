import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test

_INPUT_SHAPE = [2, 3, 4]


def _make_reduce_mean_model_attr(axes, keepdims, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, _INPUT_SHAPE)
    out_info = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [2, 1, 4])
    node = helper.make_node('ReduceMean', inputs=['data'], outputs=['reduced'],
                            axes=axes, keepdims=keepdims)
    return make_single_op_model('ReduceMean', [node], [data_info], [out_info], opset_version)


def _make_reduce_mean_model_input(axes, keepdims, opset_version):
    out_shape = [s if i not in axes else 1 for i, s in enumerate(_INPUT_SHAPE)]
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, _INPUT_SHAPE)
    out_info = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, out_shape)
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


@pytest.mark.parametrize('axes', ([0], [1]))
@pytest.mark.parametrize('opset_version', [18])
def test_reduce_mean_axes_input(axes, opset_version):
    data = np.random.randn(2, 3, 4).astype(np.float32)
    model = _make_reduce_mean_model_input(axes, 1, opset_version)
    run_op_test(model, {'data': data}, ['data'])
