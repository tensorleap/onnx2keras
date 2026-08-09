from .customonnxlayer.onnxscattertodense import TLScatterToDense
from .customonnxlayer.onnxsparseconv import TLSparseConv3DCoords, TLSparseConv3DFeatures


def convert_tl_sparse_conv3d(node, params, layers, lambda_func, node_name, keras_names):
    """Convert a TLSparseConv3D custom op node (our own export of a spconv-style
    3D sparse convolution -- see onnxsparseconv) into Keras.

    Two inputs (coords, feats) and two ONNX outputs (coords, feats), emitted as
    two SINGLE-output Keras layers: a multi-output layer breaks
    keras_data_format_converter, which assumes one output per layer when it
    walks the graph.
    """
    in_coords = layers[node.input[0]]
    in_feats = layers[node.input[1]]
    weight = layers[node.input[2]]
    bias = layers[node.input[3]]

    geometry = dict(
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        padding=params["padding"],
        dilation=params.get("dilation", [1, 1, 1]),
        in_shape=params["in_shape"],
        subm=bool(params.get("subm", 0)),
    )
    base_name = params.get("cleaned_name", node_name)

    out_coords = TLSparseConv3DCoords(name=f"{base_name}_coords", **geometry)(in_coords)
    out_feats = TLSparseConv3DFeatures(
        in_channels=weight.shape[-2],
        out_channels=weight.shape[-1],
        weight=weight,
        bias=bias,
        name=f"{base_name}_feats",
        **geometry,
    )([in_coords, in_feats, out_coords])

    outputs = params["_outputs"]
    layers[outputs[0]] = out_coords
    layers[outputs[1]] = out_feats


def convert_tl_scatter_to_dense(node, params, layers, lambda_func, node_name, keras_names):
    """Convert a TLScatterToDense custom op node into a Keras layer call.
    Two inputs (indices, updates), one dense output. The zero-filled target is
    allocated inside the layer rather than passed as a graph constant -- see
    onnxscattertodense.TLScatterToDense for why.
    """
    indices = layers[node.input[0]]
    updates = layers[node.input[1]]

    reduction = params.get("reduction", b"update")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8")

    layer = TLScatterToDense(
        dense_shape=params["dense_shape"],
        reduction=reduction,
        name=params.get("cleaned_name", node_name),
    )
    layers[node_name] = layer([indices, updates])
