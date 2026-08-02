from .customonnxlayer.onnxsparseconv import TLSparseConv3DLayer


def convert_tl_sparse_conv3d(node, params, layers, lambda_func, node_name, keras_names):
    """Convert a TLSparseConv3D custom op node (our own export of a spconv-style
    3D sparse convolution -- see onnxsparseconv.TLSparseConv3DLayer) into a
    Keras layer call. Two inputs (coords, feats), two outputs (coords, feats).
    """
    in_coords = layers[node.input[0]]
    in_feats = layers[node.input[1]]
    weight = layers[node.input[2]]
    bias = layers[node.input[3]]

    layer = TLSparseConv3DLayer(
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        padding=params["padding"],
        dilation=params.get("dilation", [1, 1, 1]),
        in_shape=params["in_shape"],
        in_channels=weight.shape[-2],
        out_channels=weight.shape[-1],
        subm=bool(params.get("subm", 0)),
        weight=weight,
        bias=bias,
        name=params.get("cleaned_name", node_name),
    )
    out_coords, out_feats = layer([in_coords, in_feats])

    outputs = params["_outputs"]
    layers[outputs[0]] = out_coords
    layers[outputs[1]] = out_feats
