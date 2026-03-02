import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_range():
    start = np.array(0.0, dtype=np.float32)
    limit = np.array(5.0, dtype=np.float32)
    delta = np.array(1.0, dtype=np.float32)

    node = helper.make_node(
        "Range",
        inputs=["start", "limit", "delta"],
        outputs=["out"],
    )

    input_vi = [
        helper.make_tensor_value_info("start", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("limit", TensorProto.FLOAT, []),
        helper.make_tensor_value_info("delta", TensorProto.FLOAT, []),
    ]
    output_vi = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [None])]

    model = make_single_op_model("Range", [node], input_vi, output_vi, opset_version=11)
    run_op_test(
        model,
        {"start": start, "limit": limit, "delta": delta},
        ["start", "limit", "delta"],
    )
