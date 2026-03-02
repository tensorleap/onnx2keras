import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
from onnx import helper, TensorProto
from onnx2kerastl import onnx_to_keras

NUMPY_DTYPE_TO_PROTO = {
    np.dtype('float32'): TensorProto.FLOAT,
    np.dtype('float64'): TensorProto.DOUBLE,
    np.dtype('int32'):   TensorProto.INT32,
    np.dtype('int64'):   TensorProto.INT64,
    np.dtype('bool'):    TensorProto.BOOL,
    np.dtype('uint8'):   TensorProto.UINT8,
    np.dtype('int8'):    TensorProto.INT8,
    np.dtype('float16'): TensorProto.FLOAT16,
}


def make_single_op_model(op_type, nodes, input_value_infos, output_value_infos, opset_version, initializers=None):
    graph = helper.make_graph(
        nodes,
        f"test_{op_type}",
        input_value_infos,
        output_value_infos,
        initializer=initializers or [],
    )
    model = helper.make_model(graph)
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = opset_version
    onnx.checker.check_model(model)
    return model


def run_op_test(onnx_model, input_dict, input_names, atol=1e-5, rtol=1e-4):
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    ort_outputs = sess.run(None, input_dict)

    keras_model = onnx_to_keras(
        onnx_model, input_names, name_policy='attach_weights_name'
    ).converted_model

    if len(input_names) == 1:
        keras_input = tf.constant(input_dict[input_names[0]])
        keras_outputs = keras_model(keras_input)
    else:
        keras_inputs = [tf.constant(input_dict[n]) for n in input_names]
        keras_outputs = keras_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    op_type = onnx_model.graph.node[0].op_type
    for i, (ort_out, keras_out) in enumerate(zip(ort_outputs, keras_outputs)):
        np.testing.assert_allclose(
            ort_out,
            keras_out.numpy(),
            atol=atol,
            rtol=rtol,
            err_msg=f"{op_type} output[{i}] mismatch",
        )
