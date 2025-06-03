import os.path

import onnx
import pytest
import tensorflow as tf
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.onnx import FeaturesManager
from pathlib import Path
from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
import torch
import subprocess

def export_llama_to_onnx(model_name, model_output_name="onnx_model"):
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", f"{model_name}",
        "--task", "causal-lm",
        f"{model_output_name}"
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Export succeeded.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Export failed.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("❌ Error: 'optimum-cli' command not found. Did you install Optimum?")
    except Exception as ex:
        print("❌ Unexpected error:", str(ex))

# @pytest.mark.skip(reason="Fails on CI but works locally (might be too big?)")
def test_llama():
    # onnx_remote_path = '/Users/nirbenzikri/PycharmProjects/PythonProject/onnx/llama_onnx_1b_instruct/model.onnx'
    onnx_model_folder = 'onnx_model'
    onnx_path = os.path.join(onnx_model_folder, 'model.onnx')
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # --------------------------------- Export to ONNX -------------------------------------
    export_llama_to_onnx(model_name, onnx_model_folder)
    # ----------------------------------------- Input Preparation --------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    text = "i love this movie!"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"What is the sentiment of this sentence: \"{text}\"? Respond with 'positive' or 'negative' only."}],
        add_generation_prompt=True,
        return_tensors="np"
    )
    input_ids = prompt
    attention_mask = (input_ids != tokenizer.pad_token_id).astype(np.int64)
    position_ids = np.arange(input_ids.shape[1])[None, :]
    model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
    keras_inputs = {k: tf.convert_to_tensor(v) for k, v in model_inputs.items()}
    # --------------------------------- Export to Keras -------------------------------------
    onnx_model = onnx.load(onnx_path)  # TODO: add to requirements, updated onnx==1.17.0 ()
    keras_model = onnx_to_keras(onnx_model, ['input_ids', 'attention_mask', 'position_ids'],
                                # input_types=[tf.int64, tf.int64, tf.float64],
                                allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    flipped_model = convert_channels_first_to_last(keras_model, [])
    flipped_model.save('temp.h5')
    model = tf.keras.models.load_model('temp.h5', custom_objects=onnx_custom_objects_map)
    # --------------------------------- Evaluating Inference -------------------------------------
    outputs = model(keras_inputs)
    last_token_logits = outputs[0, -1]
    pred_token_id = np.argmax(last_token_logits)
    pred_token = tokenizer.decode([pred_token_id]).strip().lower()

    assert pred_token=='positive'


def test_load_llama():
    model = tf.keras.models.load_model('temp_old_1b.h5', custom_objects=onnx_custom_objects_map)
    assert 1==1

if __name__ == "__main__":
    test_llama()