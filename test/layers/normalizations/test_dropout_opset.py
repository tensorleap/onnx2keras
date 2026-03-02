import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_dropout_inference():
    x = np.random.randn(2, 3, 4).astype(np.float32)

    node = helper.make_node(
        "Dropout",
        inputs=["x"],
        outputs=["y"],
    )

    input_vi = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])]
    output_vi = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])]

    model = make_single_op_model("Dropout", [node], input_vi, output_vi, opset_version=13)
    run_op_test(model, {"x": x}, ["x"])
