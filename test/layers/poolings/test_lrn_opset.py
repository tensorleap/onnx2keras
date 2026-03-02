import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_lrn():
    X = np.random.randn(1, 3, 8, 8).astype(np.float32)

    node = helper.make_node(
        "LRN",
        inputs=["X"],
        outputs=["Y"],
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        size=5,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 8, 8])]

    model = make_single_op_model("LRN", [node], input_vi, output_vi, opset_version=13)
    run_op_test(model, {"X": X}, ["X"])
