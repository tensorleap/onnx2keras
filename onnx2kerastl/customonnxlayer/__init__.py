from onnx2kerastl.customonnxlayer.onnxeinsum import OnnxEinsumLayer
from onnx2kerastl.customonnxlayer.onnxlstm import OnnxLSTM
from onnx2kerastl.customonnxlayer.onnxconstant import OnnxConstant
from onnx2kerastl.customonnxlayer.onnxsparseconv import TLSparseConv3DLayer
from onnx2kerastl.customonnxlayer.onnxscattertodense import TLScatterToDense

onnx_custom_layers = {
    "OnnxLSTM": OnnxLSTM,
    "OnnxEinsumLayer": OnnxEinsumLayer,
    "OnnxConstant": OnnxConstant,
    "TLSparseConv3DLayer": TLSparseConv3DLayer,
    "TLScatterToDense": TLScatterToDense
}
