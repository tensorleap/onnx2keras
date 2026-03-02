import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_conv2d():
    X = np.random.randn(1, 3, 8, 8).astype(np.float32)
    W = numpy_helper.from_array(
        np.random.randn(8, 3, 3, 3).astype(np.float32), name="W"
    )
    B = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="B")

    node = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        group=1,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 8, 8])]

    model = make_single_op_model(
        "Conv",
        [node],
        input_vi,
        output_vi,
        opset_version=11,
        initializers=[W, B],
    )
    run_op_test(model, {"X": X}, ["X"])
