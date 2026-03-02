import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_non_zero():
    x = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)

    node = helper.make_node(
        "NonZero",
        inputs=["x"],
        outputs=["nz"],
    )

    input_vi = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("nz", TensorProto.INT64, [2, None])]

    model = make_single_op_model("NonZero", [node], input_vi, output_vi, opset_version=13)
    run_op_test(model, {"x": x}, ["x"])
