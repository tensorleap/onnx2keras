import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [1])
def test_topk_k_attr(opset_version):
    X = np.random.randn(3, 5).astype(np.float32)

    node = helper.make_node(
        "TopK",
        inputs=["X"],
        outputs=["values", "indices"],
        k=2,
        axis=-1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])]
    output_vi = [
        helper.make_tensor_value_info("values", TensorProto.FLOAT, [3, 2]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2]),
    ]

    model = make_single_op_model("TopK", [node], input_vi, output_vi, opset_version)
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [10])
def test_topk_k_input(opset_version):
    X = np.random.randn(3, 5).astype(np.float32)
    k = numpy_helper.from_array(np.array([2], dtype=np.int64), name="k")

    node = helper.make_node(
        "TopK",
        inputs=["X", "k"],
        outputs=["values", "indices"],
        axis=-1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])]
    output_vi = [
        helper.make_tensor_value_info("values", TensorProto.FLOAT, [3, 2]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2]),
    ]

    model = make_single_op_model("TopK", [node], input_vi, output_vi, opset_version, initializers=[k])
    run_op_test(model, {"X": X}, ["X"])


@pytest.mark.parametrize("opset_version", [11])
def test_topk_k_input_sorted(opset_version):
    X = np.random.randn(3, 5).astype(np.float32)
    k = numpy_helper.from_array(np.array([2], dtype=np.int64), name="k")

    node = helper.make_node(
        "TopK",
        inputs=["X", "k"],
        outputs=["values", "indices"],
        largest=1,
        sorted=1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])]
    output_vi = [
        helper.make_tensor_value_info("values", TensorProto.FLOAT, [3, 2]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 2]),
    ]

    model = make_single_op_model(
        "TopK", [node], input_vi, output_vi, opset_version, initializers=[k]
    )
    run_op_test(model, {"X": X}, ["X"])
