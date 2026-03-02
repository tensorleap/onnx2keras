import numpy as np
import pytest
from onnx import helper, TensorProto
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [6, 9, 13])
def test_cast_to_float(opset_version):
    X = np.random.randint(0, 100, size=(3, 4)).astype(np.int64)

    node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["Y"],
        to=TensorProto.FLOAT,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.INT64, [3, 4])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])]

    model = make_single_op_model("Cast", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [6, 13])
def test_cast_to_int64(opset_version):
    X = np.random.randn(3, 4).astype(np.float32)

    node = helper.make_node(
        "Cast",
        inputs=["X"],
        outputs=["Y"],
        to=TensorProto.INT64,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.INT64, [3, 4])]

    model = make_single_op_model("Cast", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])
