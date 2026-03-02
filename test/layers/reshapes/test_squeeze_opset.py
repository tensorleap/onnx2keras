import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_squeeze_model_attr(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 1, 4])
    out_shape = [d for i, d in enumerate([1, 3, 1, 4]) if i not in axes]
    out_info = helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, out_shape)
    node = helper.make_node('Squeeze', inputs=['data'], outputs=['squeezed'], axes=axes)
    return make_single_op_model('Squeeze', [node], [data_info], [out_info], opset_version)


def _make_squeeze_model_input(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 1, 4])
    out_shape = [d for i, d in enumerate([1, 3, 1, 4]) if i not in axes]
    out_info = helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, out_shape)
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name='axes')
    node = helper.make_node('Squeeze', inputs=['data', 'axes'], outputs=['squeezed'])
    return make_single_op_model('Squeeze', [node], [data_info], [out_info], opset_version,
                                initializers=[axes_init])


@pytest.mark.parametrize('opset_version', [1, 11])
def test_squeeze_axes_as_attribute(opset_version):
    data = np.random.randn(1, 3, 1, 4).astype(np.float32)
    model = _make_squeeze_model_attr([0, 2], opset_version)
    run_op_test(model, {'data': data}, ['data'])


@pytest.mark.parametrize('opset_version', [13])
def test_squeeze_axes_as_input(opset_version):
    data = np.random.randn(1, 3, 1, 4).astype(np.float32)
    model = _make_squeeze_model_input([0, 2], opset_version)
    run_op_test(model, {'data': data}, ['data'])
