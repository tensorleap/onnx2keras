import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_global_maxpool():
    X = np.random.randn(1, 3, 8, 8).astype(np.float32)

    node = helper.make_node(
        "GlobalMaxPool",
        inputs=["X"],
        outputs=["Y"],
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 1, 1])]

    model = make_single_op_model("GlobalMaxPool", [node], input_vi, output_vi, opset_version=1)
    run_op_test(model, {"X": X}, ["X"])
