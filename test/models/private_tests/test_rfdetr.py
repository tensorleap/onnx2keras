import numpy as np
import onnx
import onnxruntime as ort
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras

ONNX_PATH = "rfdetr-base.onnx"

import tensorflow as tf
def test_rfdetr_base_local():
    onnx_model = onnx.load(ONNX_PATH)

    rng = np.random.default_rng(seed=42)
    session = ort.InferenceSession(ONNX_PATH)
    input_infos = session.get_inputs()
    output_infos = session.get_outputs()

    input_names = [info.name for info in input_infos]
    input_arrays = {info.name: rng.random(info.shape).astype(np.float32) for info in input_infos}
    output_names = [info.name for info in output_infos]

    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model
    final_model = convert_channels_first_to_last(
        keras_model, should_transform_inputs_and_outputs=False
    )

    final_model.save('yipi.h5')
    reraise = tf.keras.models.load_model('yipi.h5')
    onnx_outputs = session.run(output_names, input_feed=input_arrays)
    keras_inputs = [input_arrays[name] for name in input_names]
    keras_outputs = final_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    assert len(keras_outputs) == len(onnx_outputs)

    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out
        diff = np.abs(keras_np - onnx_out)
        assert diff.mean() < 1e-4, f"Output {i} mean error {diff.mean()} exceeds threshold"
        assert diff.max() < 1e-3, f"Output {i} max error {diff.max()} exceeds threshold"


if __name__ == "__main__":
    test_rfdetr_base_local()
