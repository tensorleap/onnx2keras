import torch
from test.utils import convert_and_test
import numpy as np
import pytest
import io
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import itertools


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, bidirectional=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.gru_layer = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                bias=bias, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, input_in, hidden_layer):
        return self.gru_layer(input_in, hidden_layer)


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('batch_first', [False, True])
@pytest.mark.parametrize('bidirectional', [False, True])
def test_rnn(bias, batch_first, bidirectional):
    input_size = 10
    hidden_size = 20
    num_layers = 2
    pt_model = RNN(input_size, hidden_size, num_layers, bias, batch_first, bidirectional)
    if batch_first is False:
        input_t = torch.randn(5, 3, 10)
    else:
        input_t = torch.randn(3, 5, 10)
    bidirectional_factor = 2 if bidirectional else 1
    h0_t = torch.randn(2*bidirectional_factor, 3, 20)
    # error = convert_and_test(pt_model, (input_t.numpy(), h0_t.numpy()), verbose=False, change_ordering=False)

    temp_f = io.BytesIO()
    torch.onnx.export(pt_model, (input_t, h0_t), temp_f, verbose=True,
                      input_names=['test_in1', 'test_in2'],
                      output_names=['test_out1', 'test_out2'])
    temp_f.seek(0)
    onnx_model = onnx.load(temp_f)
    keras_model = onnx_to_keras(onnx_model, ['test_in1', 'test_in2'], name_policy='attach_weights_name')
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_res = final_model([input_t.numpy().swapaxes(1, 2), h0_t.numpy().swapaxes(1, 2)])
    pt_res = pt_model(input_t, h0_t)
    diff_tens_state = (pt_res[0].swapaxes(1, 2).detach().numpy() - keras_res[0]).numpy().__abs__()
    diff_tens_out = (pt_res[1].swapaxes(1, 2).detach().numpy() - keras_res[1]).numpy().__abs__()
    eps = 10**(-5)
    if (diff_tens_state.max() < eps) & (diff_tens_out.max() < eps) is False:
        print(1)
    assert (diff_tens_state.max() < eps) & (diff_tens_out.max() < eps)
