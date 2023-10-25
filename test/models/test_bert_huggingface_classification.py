import onnx
import pytest
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertModel
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from transformers import AutoModelForSequenceClassification


# @pytest.mark.skip(reason="Fails on CI but works locally (might be too big?)")
def test_bert_huggingface_classifcation():
    onnx_path = 'bert_huggingface.onnx'
    model_name = "bert-base-uncased"
    # model_name_for_features = "bert"
    # id2label = {0: "IS_DAMAGED", 1: "NOT_DAMAGED"}
    # label2id = {"IS_DAMAGED": 0, "NOT_DAMAGED": 1}
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    real_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt" , padding='max_length', max_length=512)
    # OnnxConfig.default_fixed_sequence = 8  # this does nothing here, serves as a reminder
    # OnnxConfig.default_fixed_batch = 2  # this does nothing here, serves as a reminder
    # albert_features = list(FeaturesManager.get_supported_features_for_model_type(model_name_for_features).keys())
    # print(albert_features)
    # onnx_path = Path(onnx_path)
    # model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='sequence-classification')
    # onnx_config = model_onnx_config(model.config)
    # onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
    onnx_model = onnx.load(onnx_path)
    # shape_info = onnx.shape_inference.infer_shapes(onnx_model)
    # value_info_protos = []
    # inter_layers= ['onnx::Add_312']
    # for idx, node in enumerate(shape_info.graph.value_info):
    #     if node.name in inter_layers:
    #         print(idx, node)
    #         value_info_protos.append(node)
    # onnx_model.graph.output.extend(value_info_protos)  # in inference stage, these tensor will be added to output dict.
    # from onnx import helper
    #
    # # Load the ONNX model
    # # Create a new output node for the "LAYER" tensor
    # output_name = 'onnx::MatMul_310'  # Replace with the actual name of your intermediate layer
    #
    # # Find the node that produces the "LAYER" tensor
    # found = False
    # for node in onnx_model.graph.node:
    #     if output_name in node.output:
    #         # Add the "LAYER" tensor as an output
    #         onnx_model.graph.output.append(helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape=[None, None, None]))  # Fill in the correct shape
    #
    #         # Update the existing node to produce "LAYER" as its output
    #         # node.output.remove(output_name)
    #         found = True
    #         break
    #
    # if not found:
    #     print(f"Layer '{output_name}' not found in the model.")
    # input_np = [real_inputs['input_ids'].numpy(),
    #             real_inputs['token_type_ids'].numpy(),
    #             real_inputs['attention_mask'].numpy()]
    # # Save the modified model
    # onnx.checker.check_model(onnx_model)
    # import onnxruntime as rt
    # sess = rt.InferenceSession(onnx_model.SerializeToString())
    # input_name_1 = sess.get_inputs()[0].name
    # input_name_2 = sess.get_inputs()[1].name
    # input_name_3 = sess.get_inputs()[2].name
    # label_name = sess.get_outputs()[1].name
    # pred = sess.run([label_name], {input_name_1: input_np[0], input_name_2: input_np[1], input_name_3: input_np[2]})[0]
    # onnx.save(onnx_model, 'modified_model.onnx')
    #
    # onnx.save(onnx_model, 'b.onnx')
    # onnx_model = onnx.load(onnx_path)
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'token_type_ids', 'attention_mask'],
                                input_types=[tf.int32, tf.int32, tf.float32])
    input_np = [real_inputs['input_ids'].numpy(),
                real_inputs['token_type_ids'].numpy(),
                real_inputs['attention_mask'].numpy()]
    # with torch.no_grad():
    #     out = model(**real_inputs)
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_otpt = flipped_model(input_np)
    print(flipped_otpt)
    assert np.abs((out['logits'].detach().numpy() - flipped_otpt[0])).max() < 1e-04
