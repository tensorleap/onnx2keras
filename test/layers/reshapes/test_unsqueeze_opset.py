import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_unsqueeze_model_attr(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('unsqueezed', TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node('Unsqueeze', inputs=['data'], outputs=['unsqueezed'], axes=axes)
    return make_single_op_model('Unsqueeze', [node], [data_info], [out_info], opset_version)


def _make_unsqueeze_model_input(axes, opset_version):
    data_info = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('unsqueezed', TensorProto.FLOAT, [1, 3, 4])
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name='axes')
    node = helper.make_node('Unsqueeze', inputs=['data', 'axes'], outputs=['unsqueezed'])
    return make_single_op_model('Unsqueeze', [node], [data_info], [out_info], opset_version,
                                initializers=[axes_init])


@pytest.mark.parametrize('opset_version', [1, 11])
def test_unsqueeze_axes_as_attribute(opset_version):
    data = np.random.randn(3, 4).astype(np.float32)
    model = _make_unsqueeze_model_attr([0], opset_version)
    run_op_test(model, {'data': data}, ['data'])


@pytest.mark.parametrize('opset_version', [13])
def test_unsqueeze_axes_as_input(opset_version):
    data = np.random.randn(3, 4).astype(np.float32)
    model = _make_unsqueeze_model_input([0], opset_version)
    run_op_test(model, {'data': data}, ['data'])
