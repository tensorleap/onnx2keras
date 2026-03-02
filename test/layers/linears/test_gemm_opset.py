import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_gemm():
    A = np.random.randn(2, 3).astype(np.float32)
    B = numpy_helper.from_array(np.random.randn(3, 4).astype(np.float32), name="B")
    C = numpy_helper.from_array(np.random.randn(4).astype(np.float32), name="C")

    node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        transA=0,
        transB=0,
        alpha=1.0,
        beta=1.0,
    )

    input_vi = [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])]

    model = make_single_op_model(
        "Gemm",
        [node],
        input_vi,
        output_vi,
        opset_version=13,
        initializers=[B, C],
    )
    run_op_test(model, {"A": A}, ["A"])
