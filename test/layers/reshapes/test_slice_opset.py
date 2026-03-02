import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper

from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def _make_slice_model_attr(starts, ends, axes, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('sliced', TensorProto.FLOAT, [3, 2])
    node = helper.make_node('Slice', inputs=['x'], outputs=['sliced'],
                            starts=starts, ends=ends, axes=axes)
    return make_single_op_model('Slice', [node], [x_info], [out_info], opset_version)


def _make_slice_model_input(starts, ends, axes, opset_version):
    x_info = helper.make_tensor_value_info('x', TensorProto.FLOAT, [3, 4])
    out_info = helper.make_tensor_value_info('sliced', TensorProto.FLOAT, [3, 2])
    starts_init = numpy_helper.from_array(np.array(starts, dtype=np.int64), name='starts')
    ends_init = numpy_helper.from_array(np.array(ends, dtype=np.int64), name='ends')
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name='axes')
    node = helper.make_node('Slice', inputs=['x', 'starts', 'ends', 'axes'], outputs=['sliced'])
    return make_single_op_model('Slice', [node], [x_info], [out_info], opset_version,
                                initializers=[starts_init, ends_init, axes_init])


@pytest.mark.parametrize('opset_version', [1])
def test_slice_attr(opset_version):
    x = np.random.randn(3, 4).astype(np.float32)
    model = _make_slice_model_attr([0], [2], [1], opset_version)
    run_op_test(model, {'x': x}, ['x'])


@pytest.mark.parametrize('opset_version', [10, 13])
def test_slice_input(opset_version):
    x = np.random.randn(3, 4).astype(np.float32)
    model = _make_slice_model_input([0], [2], [1], opset_version)
    run_op_test(model, {'x': x}, ['x'])
