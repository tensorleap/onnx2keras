import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [2, 11])
def test_split_attr(opset_version):
    X = np.random.randn(2, 4).astype(np.float32)

    node = helper.make_node(
        "Split",
        inputs=["X"],
        outputs=["out0", "out1"],
        axis=1,
        split=[2, 2],
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])]
    output_vi = [
        helper.make_tensor_value_info("out0", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("out1", TensorProto.FLOAT, [2, 2]),
    ]

    model = make_single_op_model("Split", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [13])
def test_split_input(opset_version):
    X = np.random.randn(2, 4).astype(np.float32)
    split_sizes = numpy_helper.from_array(
        np.array([2, 2], dtype=np.int64), name="split_sizes"
    )

    node = helper.make_node(
        "Split",
        inputs=["X", "split_sizes"],
        outputs=["out0", "out1"],
        axis=1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])]
    output_vi = [
        helper.make_tensor_value_info("out0", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("out1", TensorProto.FLOAT, [2, 2]),
    ]

    model = make_single_op_model(
        "Split", [node], input_vi, output_vi, opset_version, initializers=[split_sizes]
    )
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [18])
def test_split_num_outputs(opset_version):
    X = np.random.randn(2, 4).astype(np.float32)

    node = helper.make_node(
        "Split",
        inputs=["X"],
        outputs=["out0", "out1"],
        axis=1,
        num_outputs=2,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])]
    output_vi = [
        helper.make_tensor_value_info("out0", TensorProto.FLOAT, [2, 2]),
        helper.make_tensor_value_info("out1", TensorProto.FLOAT, [2, 2]),
    ]

    model = make_single_op_model("Split", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])
