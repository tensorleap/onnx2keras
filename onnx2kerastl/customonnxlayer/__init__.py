from onnx2kerastl.customonnxlayer.onnxless import OnnxLess
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM

onnx_custom_objects_map = {
    "OnnxLSTM": OnnxLSTM,
    "OnnxLess": OnnxLess
}
