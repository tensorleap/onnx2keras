import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import onnxruntime as ort
import torch


class wrapper_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, tensor):
        ff = self.inner_model(tensor)
        return ff


def test_dinov2():
    batch_size = 1
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    wm = wrapper_model(dinov2_vits14).to('cpu')
    wm.eval()
    dummy_input = torch.FloatTensor(np.random.uniform(0, 1, (batch_size, 3, 224, 224)))
    torch.onnx.export(wm, dummy_input, "dino-2-test.onnx", input_names=['img'],
                      output_names=['vit_out'])
    np_input = list(np.random.rand(1, 3, 224, 224))
    onnx_model = onnx.load('dino-2-test.onnx')
    keras_model = onnx_to_keras(onnx_model, ['img', 'masks'], allow_partial_compilation=False)
    flipped_model = convert_channels_first_to_last(keras_model.converted_model, should_transform_inputs_and_outputs=False)
    ort_session = ort.InferenceSession('dino-2-test.onnx')
    keras_res = flipped_model(np.array(np_input))
    res = ort_session.run(
        ['vit_out'],
        input_feed={"img": np.array(np_input).astype(np.float32)}
    )
    t_mean, t_max = (res[0]-keras_res).__abs__().numpy().mean(), (res[0]-keras_res).__abs__().numpy().max()
    assert t_mean < 5e-2
    assert t_max < 0.4
