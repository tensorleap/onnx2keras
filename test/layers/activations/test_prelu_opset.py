import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [1, 6, 7, 9, 16])
def test_prelu_same_shape(opset_version):
    X = np.random.randn(2, 3).astype(np.float32)
    slope = numpy_helper.from_array(
        np.random.randn(2, 3).astype(np.float32), name="slope"
    )

    node = helper.make_node(
        "PRelu",
        inputs=["X", "slope"],
        outputs=["Y"],
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])]

    model = make_single_op_model(
        "PRelu", [node], input_vi, output_vi, opset_version, initializers=[slope]
    )
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [7, 9, 16])
def test_prelu_broadcast(opset_version):
    X = np.random.randn(2, 3).astype(np.float32)
    slope = numpy_helper.from_array(
        np.random.randn(1).astype(np.float32), name="slope"
    )

    node = helper.make_node(
        "PRelu",
        inputs=["X", "slope"],
        outputs=["Y"],
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])]

    model = make_single_op_model(
        "PRelu", [node], input_vi, output_vi, opset_version, initializers=[slope]
    )
    run_op_test(model, {"X": X}, ["X"])
