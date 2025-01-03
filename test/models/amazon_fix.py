import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import pytest
import onnx
from onnx import helper
import onnxruntime as rt


def add_node_to_onnx_output(node_name_list, new_name):
    # Load the ONNX model
    model_path = f'/Users/tomkoren/Downloads/gengraspv2_4m_opset14.onnx'
    onnx_model = onnx.load(model_path)
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(onnx_model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name in node_name_list:
            value_info_protos.append(node)
    assert len(value_info_protos) == len(node_name_list)
    onnx_model.graph.output.extend(value_info_protos)  # in inference stage, these tensor will be added to output dict.
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, new_name)

add_node_to_onnx_output(
                         ['/visual_encoder/Expand_1_output_0'], 'custom.onnx')

import onnx
from onnx import helper, TensorProto

def add_dynamic_shape_output(model_path, output_name, rank, new_name, t_type=TensorProto.INT64):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    # Create a dynamic shape for the given rank
    dynamic_shape = [f"dim_{i}" for i in range(rank)]

    # Create a new output with the dynamic shape
    new_output = helper.make_tensor_value_info(
        output_name,  # Name of the new output
        t_type,  # Data type (adjust as needed)
        dynamic_shape  # Dynamic shape
    )

    # Add the new output to the graph
    onnx_model.graph.output.append(new_output)
    onnx.checker.check_model(onnx_model)

    # Save the modified model
    onnx.save(onnx_model, new_name)
    print(f"Modified model saved as {new_name}")

add_dynamic_shape_output(f'/Users/tomkoren/Downloads/gengraspv2_4m_opset14.onnx',
                         '/visual_encoder/Expand_1_output_0',1, 'custom.onnx')
sess = rt.InferenceSession('custom.onnx')
input_name_1 = sess.get_inputs()[0].name
input_name_2 = sess.get_inputs()[1].name
label_name_1 = sess.get_outputs()[0].name
label_name_2 = sess.get_outputs()[1].name
label_name_3 = sess.get_outputs()[2].name
label_name_4 = sess.get_outputs()[3].name



pred = sess.run([label_name_1, label_name_2, label_name_3, label_name_4],
                {input_name_1: np.ones((1,3,512,512)).astype(np.float32), input_name_2: np.random.random((1,13)).astype(np.float32)})
print(1)
def ttt_amazonv2():
    model_path = f'/Users/tomkoren/Downloads/gengraspv2_4m_opset14.onnx'
    model = onnx.load(model_path)

    # Get the graph from the model
    graph = model.graph

    # Identify the node
    node_name = "/visual_encoder/Where_2"
    node = None
    for n in graph.node:
        if n.name == node_name:
            node = n
            break

    if node is None:
        raise ValueError(f"Node '{node_name}' not found in the model.")

    # Add the inputs of the node to the model's outputs
    for input_name in node.input:
        # Check if the input is already an output
        if input_name not in [o.name for o in graph.output]:
            # Get the corresponding value info from the model's initializers or inputs
            value_info = None
            for vi in graph.value_info:
                if vi.name == input_name:
                    value_info = vi
                    break
            if value_info is None:
                for inp in graph.input:
                    if inp.name == input_name:
                        value_info = inp
                        break
            if value_info is None:
                raise ValueError(f"Input '{input_name}' not found in the model's inputs or value_info.")

            # Add a new output
            new_output = helper.make_tensor_value_info(value_info.name, value_info.type.tensor_type.elem_type, None)
            graph.output.append(new_output)

    # Save the modified model
    modified_model_path = "modified_model.onnx"
    onnx.save(model, modified_model_path)

    print(f"Modified model saved to {modified_model_path}")


ttt_amazonv2()