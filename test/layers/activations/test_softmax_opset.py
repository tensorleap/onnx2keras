import numpy as np
import pytest
from onnx import helper, TensorProto
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [1, 11])
def test_softmax_axis_default_opset1_11(opset_version):
    X = np.random.randn(2, 3).astype(np.float32)

    node = helper.make_node(
        "Softmax",
        inputs=["X"],
        outputs=["Y"],
        axis=1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])]

    model = make_single_op_model("Softmax", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [13])
def test_softmax_axis_default_opset13(opset_version):
    X = np.random.randn(2, 3).astype(np.float32)

    node = helper.make_node(
        "Softmax",
        inputs=["X"],
        outputs=["Y"],
        axis=-1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])]

    model = make_single_op_model("Softmax", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])
