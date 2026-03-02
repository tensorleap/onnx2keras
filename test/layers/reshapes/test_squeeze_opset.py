import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test

_IN_SHAPE = [1, 3, 1, 4]


def _out_shape(axes):
    return [d for i, d in enumerate(_IN_SHAPE) if i not in axes]


def _make_squeeze_model_attr(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, _IN_SHAPE)
    out_info = helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, _out_shape(axes))
    node = helper.make_node('Squeeze', inputs=['data'], outputs=['squeezed'], axes=axes)
    return make_single_op_model('Squeeze', [node], [data_info], [out_info], opset_version)


def _make_squeeze_model_input(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, _IN_SHAPE)
    out_info = helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, _out_shape(axes))
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name='axes')
    node = helper.make_node('Squeeze', inputs=['data', 'axes'], outputs=['squeezed'])
    return make_single_op_model('Squeeze', [node], [data_info], [out_info], opset_version,
                                initializers=[axes_init])


@pytest.mark.parametrize('opset_version', [1, 11])
def test_squeeze_axes_as_attribute(opset_version):
    data = np.random.randn(*_IN_SHAPE).astype(np.float32)
    model = _make_squeeze_model_attr([0, 2], opset_version)
    run_op_test(model, {'data': data}, ['data'])


@pytest.mark.parametrize('axes', ([0], [2], [0, 2]))
@pytest.mark.parametrize('opset_version', [13])
def test_squeeze_axes_as_input(axes, opset_version):
    data = np.random.randn(*_IN_SHAPE).astype(np.float32)
    model = _make_squeeze_model_input(axes, opset_version)
    run_op_test(model, {'data': data}, ['data'])
