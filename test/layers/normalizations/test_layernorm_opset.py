import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


def test_layernorm():
    X = np.random.randn(2, 4, 8).astype(np.float32)
    scale = numpy_helper.from_array(np.ones(8, dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bias")

    node = helper.make_node(
        "LayerNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        axis=-1,
        epsilon=1e-5,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 8])]

    model = make_single_op_model(
        "LayerNormalization",
        [node],
        input_vi,
        output_vi,
        opset_version=17,
        initializers=[scale, bias],
    )
    run_op_test(model, {"X": X}, ["X"])
