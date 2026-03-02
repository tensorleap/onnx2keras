import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_avgpool_2d():
    X = np.random.randn(1, 3, 8, 8).astype(np.float32)

    node = helper.make_node(
        "AveragePool",
        inputs=["X"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])]

    model = make_single_op_model("AveragePool", [node], input_vi, output_vi, opset_version=11)
    run_op_test(model, {"X": X}, ["X"])
