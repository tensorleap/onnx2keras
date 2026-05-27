from onnx2kerastl.customonnxlayer.onnxeinsum import OnnxEinsumLayer
from onnx2kerastl.customonnxlayer.onnxgridsample import OnnxGridSampleLayer
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM

onnx_custom_layers = {
    "OnnxLSTM": OnnxLSTM,
    "OnnxEinsumLayer": OnnxEinsumLayer,
    "OnnxGridSampleLayer": OnnxGridSampleLayer,
}
