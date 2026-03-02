import numpy as np
import pytest
from onnx import helper, TensorProto, numpy_helper
from test.layers.onnx_op_test_utils import make_single_op_model, run_op_test


@pytest.mark.parametrize("opset_version", [11, 13])
def test_scatter_nd_no_reduction(opset_version):
    data = np.random.randn(4, 3).astype(np.float32)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.random.randn(2, 3).astype(np.float32)

    indices_init = numpy_helper.from_array(indices, name="indices")
    updates_init = numpy_helper.from_array(updates, name="updates")

    node = helper.make_node(
        "ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["Y"],
    )

    input_vi = [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])]

    model = make_single_op_model(
        "ScatterND",
        [node],
        input_vi,
        output_vi,
        opset_version,
        initializers=[indices_init, updates_init],
    )
    run_op_test(model, {"data": data}, ["data"])


@pytest.mark.parametrize("opset_version", [16])
def test_scatter_nd_add_reduction(opset_version):
    data = np.random.randn(4, 3).astype(np.float32)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.random.randn(2, 3).astype(np.float32)

    indices_init = numpy_helper.from_array(indices, name="indices")
    updates_init = numpy_helper.from_array(updates, name="updates")

    node = helper.make_node(
        "ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["Y"],
        reduction="add",
    )

    input_vi = [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3])]
    output_vi = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 3])]

    model = make_single_op_model(
        "ScatterND",
        [node],
        input_vi,
        output_vi,
        opset_version,
        initializers=[indices_init, updates_init],
    )
    run_op_test(model, {"data": data}, ["data"])
