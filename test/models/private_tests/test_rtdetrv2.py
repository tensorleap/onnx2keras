from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras
from test.models.private_tests.aws_utils import aws_s3_download

INT_DTYPES = {
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def _normalize_shape(shape):
    normalized = []
    for dim in shape:
        if isinstance(dim, str) or dim is None:
            normalized.append(1)
        elif isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return tuple(normalized)


def _build_random_input(input_info, rng):
    shape = _normalize_shape(input_info.shape)

    if input_info.type == "tensor(bool)":
        return rng.integers(0, 2, size=shape).astype(bool)

    if input_info.type in INT_DTYPES:
        return rng.integers(0, 10, size=shape).astype(INT_DTYPES[input_info.type])

    return rng.random(shape).astype(np.float32)

@pytest.mark.parametrize('aws_s3_download', [["rtdetrv2/", "rtdetrv2/", False]], indirect=True)
def test_rtdetrv2_local(aws_s3_download):
    onnx_path = f'{aws_s3_download}/rtdetrv2_r18vd_120e_raw_outputs.onnx'
    onnx_model = onnx.load(onnx_path)

    rng = np.random.default_rng(seed=42)
    session = ort.InferenceSession(str(onnx_path))
    input_infos = session.get_inputs()
    output_infos = session.get_outputs()

    input_names = [info.name for info in input_infos]
    input_arrays = {info.name: _build_random_input(info, rng) for info in input_infos}
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

    onnx_outputs = session.run(output_names, input_feed=input_arrays)
    keras_inputs = [input_arrays[name] for name in input_names]
    keras_outputs = final_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    assert len(keras_outputs) == len(onnx_outputs)

    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out
        if onnx_out.dtype == bool or keras_np.dtype == bool:
            assert np.array_equal(keras_np, onnx_out), f"Output {i} boolean mismatch"
            continue

        if np.issubdtype(onnx_out.dtype, np.integer) and np.issubdtype(
            keras_np.dtype, np.integer
        ):
            assert np.array_equal(keras_np, onnx_out), f"Output {i} integer mismatch"
            continue

        diff = np.abs(keras_np - onnx_out)
        mean_error = diff.mean()
        max_error = diff.max()
        assert mean_error < 1e-4, f"Output {i} mean error {mean_error} exceeds threshold"
        assert max_error < 1e-3, f"Output {i} max error {max_error} exceeds threshold"


@pytest.mark.skip(reason="model has float64/float32 type mismatch in Where op — needs fix before running")
def test_rtdetr_client_local():
    # TODO: ORT fails to load this model due to a type mismatch in the Where op
    # (/model/decoder/Where): one branch produces float32 (from Log) while the
    # other is a float64 constant (inf). The Where op requires both tensor inputs
    # to share the same type. The model needs to be fixed before this test can run.
    onnx_path = "/Users/ranhomri/repos/hub/RT-DETR/client_format_structure.onnx"
    onnx_model = onnx.load(onnx_path)

    rng = np.random.default_rng(seed=42)
    session = ort.InferenceSession(onnx_path)
    input_infos = session.get_inputs()
    output_infos = session.get_outputs()

    input_names = [info.name for info in input_infos]
    input_arrays = {info.name: _build_random_input(info, rng) for info in input_infos}
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

    onnx_outputs = session.run(output_names, input_feed=input_arrays)
    keras_inputs = [input_arrays[name] for name in input_names]
    keras_outputs = final_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    assert len(keras_outputs) == len(onnx_outputs)

    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out
        if onnx_out.dtype == bool or keras_np.dtype == bool:
            assert np.array_equal(keras_np, onnx_out), f"Output {i} boolean mismatch"
            continue

        if np.issubdtype(onnx_out.dtype, np.integer) and np.issubdtype(
            keras_np.dtype, np.integer
        ):
            assert np.array_equal(keras_np, onnx_out), f"Output {i} integer mismatch"
            continue

        diff = np.abs(keras_np - onnx_out)
        mean_error = diff.mean()
        max_error = diff.max()
        assert mean_error < 1e-4, f"Output {i} mean error {mean_error} exceeds threshold"
        assert max_error < 1e-3, f"Output {i} max error {max_error} exceeds threshold"

    print("valid conversion")


if __name__ == "__main__":
    test_rtdetr_client_local()
