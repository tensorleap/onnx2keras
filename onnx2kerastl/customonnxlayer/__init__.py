from onnx2kerastl.customonnxlayer.onnxeinsum import OnnxEinsumLayer
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM

onnx_custom_objects_map = {
    "OnnxLSTM": OnnxLSTM,
    "OnnxEinsumLayer": OnnxEinsumLayer
}
