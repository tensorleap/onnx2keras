"""Conversion test for the TLSparseConv3D custom op -- our own hand-built ONNX
representation of a spconv-style 3D sparse convolution (see
onnx2kerastl/customonnxlayer/onnxsparseconv.py), used as the LiDAR-backbone
building block once we export BEVFusion's own PyTorch SparseEncoder rather
than relying on NVIDIA's CUDA-BEVFusion export (which omits the voxel
coordinates entirely and can't be converted at all -- see conversation notes).

This graph is hand-built (not exported from PyTorch) because we're testing
the onnx2keras converter/layer wiring itself, not re-validating the
underlying math -- that was already verified against the real compiled
spconv CUDA op on EC2 (see scratchpad/sparse_conv_poc).

Two stacked layers mirror the real SparseEncoder's structure:
  conv0: submanifold (subm=True), kernel=3 stride=1 padding=1 -- same coords out
  conv1: regular/downsampling (subm=False), kernel=3 stride=2 padding=1
"""
import itertools
import os

import numpy as np
import onnx
import pytest
import tensorflow as tf
from onnx import TensorProto, helper

from onnx2kerastl import onnx_to_keras
from onnx2kerastl.customonnxlayer import onnx_custom_layers

HERE = os.path.dirname(os.path.abspath(__file__))


def _make_taps(kernel_size):
    return list(itertools.product(*[range(k) for k in kernel_size]))


def _naive_sparse_conv(in_coords, in_feats, kernel, bias, kernel_size, stride, padding, dilation, in_shape, subm):
    taps = _make_taps(kernel_size)
    in_coord_set = {tuple(c): i for i, c in enumerate(in_coords)}

    if subm:
        out_shape = in_shape
        active_out = [tuple(c) for c in in_coords]
    else:
        out_shape = [
            (in_shape[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) // stride[d] + 1
            for d in range(3)
        ]
        active_out = []
        for oz in range(out_shape[0]):
            for oy in range(out_shape[1]):
                for ox in range(out_shape[2]):
                    o = np.array([oz, oy, ox])
                    for t in taps:
                        p = o * np.array(stride) - np.array(padding) + np.array(t) * np.array(dilation)
                        if tuple(p) in in_coord_set:
                            active_out.append(tuple(o))
                            break

    active_out = np.array(active_out, dtype=np.int64) if active_out else np.zeros((0, 3), dtype=np.int64)
    out_feats = np.zeros((len(active_out), kernel.shape[-1]), dtype=np.float32)
    for i, o in enumerate(active_out):
        acc = np.zeros(kernel.shape[-1], dtype=np.float32)
        for k, t in enumerate(taps):
            p = tuple(o * np.array(stride) - np.array(padding) + np.array(t) * np.array(dilation))
            j = in_coord_set.get(p)
            if j is not None:
                acc += in_feats[j] @ kernel[k]
        out_feats[i] = acc + bias
    return active_out, out_feats, out_shape


def _sparse_conv_node(name, in_coords_name, in_feats_name, out_coords_name, out_feats_name,
                       weight_name, bias_name, kernel_size, stride, padding, dilation, in_shape, subm):
    return helper.make_node(
        "TLSparseConv3D",
        inputs=[in_coords_name, in_feats_name, weight_name, bias_name],
        outputs=[out_coords_name, out_feats_name],
        name=name,
        domain="ai.tensorleap",
        kernel_size=list(kernel_size),
        stride=list(stride),
        padding=list(padding),
        dilation=list(dilation),
        in_shape=list(in_shape),
        subm=int(subm),
    )


def _build_case():
    rng = np.random.default_rng(7)
    in_shape0 = (8, 8, 8)
    c_in, c_mid, c_out = 4, 6, 8

    all_coords = np.array(list(itertools.product(*[range(s) for s in in_shape0])))
    chosen = rng.choice(len(all_coords), size=15, replace=False)
    in_coords = all_coords[chosen].astype(np.int64)
    in_feats = rng.standard_normal((len(in_coords), c_in)).astype(np.float32)

    # conv0: subm, kernel=3 stride=1 padding=1
    k0_size, k0_stride, k0_pad, k0_dil = (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)
    n_taps0 = int(np.prod(k0_size))
    weight0 = (rng.standard_normal((n_taps0, c_in, c_mid)) * 0.1).astype(np.float32)
    bias0 = (rng.standard_normal(c_mid) * 0.1).astype(np.float32)

    mid_coords, mid_feats, mid_shape = _naive_sparse_conv(
        in_coords, in_feats, weight0, bias0, k0_size, k0_stride, k0_pad, k0_dil, in_shape0, subm=True
    )

    # conv1: regular downsampling, kernel=3 stride=2 padding=1
    k1_size, k1_stride, k1_pad, k1_dil = (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)
    n_taps1 = int(np.prod(k1_size))
    weight1 = (rng.standard_normal((n_taps1, c_mid, c_out)) * 0.1).astype(np.float32)
    bias1 = (rng.standard_normal(c_out) * 0.1).astype(np.float32)

    out_coords, out_feats, out_shape = _naive_sparse_conv(
        mid_coords, mid_feats, weight1, bias1, k1_size, k1_stride, k1_pad, k1_dil, mid_shape, subm=False
    )

    return dict(
        in_coords=in_coords, in_feats=in_feats,
        weight0=weight0.reshape(*k0_size, c_in, c_mid), bias0=bias0,
        weight1=weight1.reshape(*k1_size, c_mid, c_out), bias1=bias1,
        in_shape0=in_shape0, mid_shape=mid_shape,
        k0=(k0_size, k0_stride, k0_pad, k0_dil), k1=(k1_size, k1_stride, k1_pad, k1_dil),
        expected_out_coords=out_coords, expected_out_feats=out_feats,
    )


def _build_onnx_model(case):
    in_shape0 = case["in_shape0"]
    mid_shape = case["mid_shape"]
    k0_size, k0_stride, k0_pad, k0_dil = case["k0"]
    k1_size, k1_stride, k1_pad, k1_dil = case["k1"]

    n_in = case["in_coords"].shape[0]
    c_in = case["in_feats"].shape[1]

    coords_in = helper.make_tensor_value_info("coords_in", TensorProto.INT64, [n_in, 3])
    feats_in = helper.make_tensor_value_info("feats_in", TensorProto.FLOAT, [n_in, c_in])

    node0 = _sparse_conv_node(
        "conv0", "coords_in", "feats_in", "mid_coords", "mid_feats",
        "weight0", "bias0", k0_size, k0_stride, k0_pad, k0_dil, in_shape0, subm=True,
    )
    node1 = _sparse_conv_node(
        "conv1", "mid_coords", "mid_feats", "out_coords", "out_feats",
        "weight1", "bias1", k1_size, k1_stride, k1_pad, k1_dil, mid_shape, subm=False,
    )

    weight0_init = helper.make_tensor("weight0", TensorProto.FLOAT, case["weight0"].shape, case["weight0"].flatten())
    bias0_init = helper.make_tensor("bias0", TensorProto.FLOAT, case["bias0"].shape, case["bias0"].flatten())
    weight1_init = helper.make_tensor("weight1", TensorProto.FLOAT, case["weight1"].shape, case["weight1"].flatten())
    bias1_init = helper.make_tensor("bias1", TensorProto.FLOAT, case["bias1"].shape, case["bias1"].flatten())

    c_out = case["weight1"].shape[-1]
    out_coords_info = helper.make_tensor_value_info("out_coords", TensorProto.INT64, ["n_out", 3])
    out_feats_info = helper.make_tensor_value_info("out_feats", TensorProto.FLOAT, ["n_out", c_out])

    graph = helper.make_graph(
        [node0, node1],
        "sparse_conv_test",
        [coords_in, feats_in],
        [out_coords_info, out_feats_info],
        initializer=[weight0_init, bias0_init, weight1_init, bias1_init],
    )
    model = helper.make_model(
        graph,
        producer_name="tl_sparse_conv_test",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("ai.tensorleap", 1)],
    )
    onnx.checker.check_model(model)
    return model


def test_sparse_conv_custom_op_conversion():
    case = _build_case()
    onnx_model = _build_onnx_model(case)

    result = onnx_to_keras(
        onnx_model,
        ["coords_in", "feats_in"],
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    )
    keras_model = result.converted_model

    h5_path = os.path.join(HERE, "test_sparse_conv_custom_op.h5")
    keras_model.save(h5_path)
    loaded_model = tf.keras.models.load_model(h5_path, custom_objects=onnx_custom_layers)

    keras_out_coords, keras_out_feats = loaded_model([case["in_coords"], case["in_feats"]])
    keras_out_coords = keras_out_coords.numpy()
    keras_out_feats = keras_out_feats.numpy()

    def flat(c, shape):
        return c[:, 0] * shape[1] * shape[2] + c[:, 1] * shape[2] + c[:, 2]

    out_shape = [
        (case["mid_shape"][d] + 2 * case["k1"][2][d] - case["k1"][3][d] * (case["k1"][0][d] - 1) - 1) // case["k1"][1][d] + 1
        for d in range(3)
    ]

    exp_order = np.argsort(flat(case["expected_out_coords"], out_shape))
    keras_order = np.argsort(flat(keras_out_coords, out_shape))

    exp_c = case["expected_out_coords"][exp_order]
    exp_f = case["expected_out_feats"][exp_order]
    ker_c = keras_out_coords[keras_order]
    ker_f = keras_out_feats[keras_order]

    assert exp_c.shape == ker_c.shape, f"coord count mismatch: expected {exp_c.shape} got {ker_c.shape}"
    assert np.array_equal(exp_c, ker_c), "active output coordinate sets differ"

    diff = np.abs(exp_f.astype(np.float64) - ker_f.astype(np.float64))
    mean_error, max_error = float(diff.mean()), float(diff.max())
    print(f"mean_error={mean_error:.8f} max_error={max_error:.8f}")
    assert max_error < 1e-3, f"feature values mismatch: max_error={max_error}"


if __name__ == "__main__":
    test_sparse_conv_custom_op_conversion()
    print("PASSED")
