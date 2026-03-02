import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [9, 14, 15])
def test_batchnorm_inference(opset_version):
    X = np.random.randn(2, 4).astype(np.float32)
    scale = numpy_helper.from_array(
        np.random.randn(4).astype(np.float32), name="scale"
    )
    bias = numpy_helper.from_array(
        np.random.randn(4).astype(np.float32), name="bias"
    )
    mean = numpy_helper.from_array(
        np.random.randn(4).astype(np.float32), name="mean"
    )
    var = numpy_helper.from_array(
        np.abs(np.random.randn(4)).astype(np.float32) + 0.1, name="var"
    )

    attrs = {}
    if opset_version >= 14:
        attrs["training_mode"] = 0

    node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "scale", "bias", "mean", "var"],
        outputs=["Y"],
        **attrs,
    )

    input_vi = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])]

    model = make_single_op_model(
        "BatchNormalization",
        [node],
        input_vi,
        output_vi,
        opset_version,
        initializers=[scale, bias, mean, var],
    )
    run_op_test(model, {"X": X}, ["X"])
