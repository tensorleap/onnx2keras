import numpy as np
import torch
import onnx
from onnx2kerastl import onnx_to_keras
import onnxruntime as ort
import tensorflow as tf
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl.customonnxlayer import onnx_custom_objects_map


def test_efficiency():
    onnx_model_path = f'detectron2/mod_efficiency.onnx'
    # save_model_path = f'effizency_models/mod_efficiency.h5'

    input_data = np.random.uniform(0, 1, (1, 3, 800, 800))
    # load onnx model
    onnx_model = onnx.load(onnx_model_path)
    # extract feature names from the model
    input_features = [inp.name for inp in onnx_model.graph.input]
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model, input_names=input_features, name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model.converted_model, should_transform_inputs_and_outputs=True, verbose=True)
    print(1)
    # final_model.save(save_model_path)
    #
    # ort_session = ort.InferenceSession(onnx_model_path)
    # onnx_outputs = ort_session.run(None, {input_features[0]: np.transpose(input_data, (0, 3, 1, 2))})
    #
    # loaded_keras_model = tf.keras.models.load_model(save_model_path, custom_objects=onnx_custom_objects_map)
    # loaded_keras_outputs = loaded_keras_model(input_data)
    #
    # onnx_pred = np.transpose(onnx_outputs[1], (0, 2, 1))
    # keras_pred = loaded_keras_outputs[1]
    #
    # is_same = np.allclose(keras_pred, onnx_pred)


#
# import onnx
# from onnx import helper, numpy_helper
# # Load the ONNX model
# onnx_model_path = 'detectron2/efficiency.onnx'
# model = onnx.load(onnx_model_path)
# # Get the input shape of the existing model
# # Get the input shape of the existing model
# input_shape = model.graph.input[0].type.tensor_type.shape.dim
#
# # Create a new input with an extra dimension on axis=0
# new_input_shape = [1] + [dim.dim_value for dim in input_shape]
# new_input_name = 'new_input'
# new_input = helper.make_tensor_value_info(new_input_name, onnx.TensorProto.FLOAT, new_input_shape)
#
# # Create a Squeeze node
# squeeze_node = helper.make_node(
#     'Squeeze',
#     inputs=[new_input_name],
#     outputs=['squeeze_output'],
#     axes=[0]  # Squeeze along axis=0
# )
#
# # Find the first node in the original model (if it exists)
# first_node_index = None
# for i, node in enumerate(model.graph.node):
#     if node.input and node.input[0] == model.graph.input[0].name:
#         first_node_index = i
#         break
#
# # Update the Squeeze node's output to be the input for the first node (if found)
# if first_node_index is not None:
#     model.graph.node[first_node_index].input[0] = 'squeeze_output'
#
#
# ####
# original_input_name = model.graph.input[0].name
#
# model.graph.input.remove(model.graph.input[0])
#
# # # Create a new list of initializers without the one associated with the original input
# # new_initializers = [init for init in model.graph.initializer if init.name != original_input_name]
# #
# # # Assign the new list back to the initializer field
# # del model.graph.initializer[:]
# # model.graph.initializer.extend(new_initializers)
#
# # Remove any nodes using the original input
# # model.graph.node[:] = [node for node in model.graph.node if original_input_name not in node.input]
#
# # Update the graph with the new input and Squeeze node
# model.graph.input.extend([new_input])
# model.graph.node.insert(0, squeeze_node)  # Insert Squeeze node at the beginning
#
#
#
#
#
#
# # Save the modified model
# onnx.save(model, 'detectron2/mod_efficiency.onnx')
#
#
#
#
#
#
#
#
#
#
# # Get the input shape of the existing model
# input_shape = model.graph.input[0].type.tensor_type.shape.dim
#
# # Create a new input with an extra dimension on axis=0
# new_input_shape = [1] + [dim.dim_value for dim in input_shape]
# new_input_name = 'new_input'
# new_input = helper.make_tensor_value_info(new_input_name, onnx.TensorProto.FLOAT, new_input_shape)
#
# # Create a Squeeze node
# squeeze_node = helper.make_node(
#     'Squeeze',
#     inputs=[new_input_name],  # Connect the new input to the Squeeze node
#     outputs=['squeezed_output'],
#     axes=[0]  # Squeeze along axis=0
# )
#
# # Update the graph with the new input and Squeeze node
# model.graph.input.extend([new_input])
# model.graph.node.insert(0, squeeze_node)  # Insert Squeeze node at the beginning
# import onnx
# import onnxruntime as rt
# import copy
# new_model = copy.deepcopy(onnx_model)
# desired_node = None
# # aa='/backbone/fpn_output2/Conv_output_0'
# # bb='/roi_heads/box_pooler/level_poolers.0/Gather_2_output_0'
# # cc='/roi_heads/box_pooler/level_poolers.0/Cast_1_output_0'
# find_name = '/roi_heads/mask_head/mask_fcn4/activation/Relu_output_0'
# for node in new_model.graph.node:
#     if find_name in node.name:
#         desired_node = node
#         break
# # Create a new output tensor with a unique name, e.g., 'new_output'
# new_output_tensor = onnx.helper.make_tensor_value_info(find_name, onnx.TensorProto.FLOAT, shape=[None])
#
# # Add the new output to the graph outputs
# new_model.graph.output.extend([new_output_tensor])
#
# # Connect the desired node's output to the new output
# new_model.graph.value_info.append(onnx.helper.make_tensor_value_info(find_name, onnx.TensorProto.FLOAT, shape=[None]))
# onnx.save(new_model, 'temp.onnx')
# sess = rt.InferenceSession('temp.onnx')
# print([sess.get_outputs()[i].name for i in range(len(sess.get_outputs()))])
# input_name_1 = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# fin = sess.run([find_name], {input_name_1: inpt.astype(np.float32)})[0]
# print(fin.shape)


def find_single_channel_conv(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Initialize a list to store CONV nodes with a single output channel
    single_channel_conv_nodes = []

    # Iterate over all nodes in the model
    for node in model.graph.node:
        # Check if the node is a CONV operation
        if node.op_type == 'Conv':
            # Find the attribute that specifies the number of output channels
            output_channels_attr = next(attr for attr in node.attribute if attr.name == 'output_channels')

            # Check if the number of output channels is 1
            if output_channels_attr.i == 1:
                # Add the node to the list
                single_channel_conv_nodes.append(node)

    return single_channel_conv_nodes


# find_single_channel_conv('/home/tomtensor/Work/Projects/tensorleap/onnx2kerras_new/onnx2keras/test/models/detectron2/mod_efficiency.onnx')