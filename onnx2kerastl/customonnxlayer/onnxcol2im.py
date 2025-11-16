import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

import tensorflow as tf


def col2im_onnx(
        input_cols,          # (N, C*kH*kW, L)   ONNX format
        image_shape,         # (H_img, W_img)
        block_shape,         # (kH, kW)
        dilations=(1, 1),
        pads=(0, 0, 0, 0),   # (pt, pl, pb, pr)
        strides=(1, 1)
):
    """
    ONNX-style Col2Im, INPUT IN ONNX FORMAT:
        input_cols: (N, C * prod(block_shape), L)
    OUTPUT (NCHW):
        (N, C, H_img, W_img)

    Behavior now matches torch.nn.Fold exactly:
      - overlap contributions are SUMMED (no normalization)
    """

    # -----------------------------
    # Unpack attributes
    # -----------------------------
    N  = tf.shape(input_cols)[0]
    Ck = tf.shape(input_cols)[1]
    L  = tf.shape(input_cols)[2]

    H_img, W_img = image_shape
    kH, kW       = block_shape
    dH, dW       = dilations
    pt, pl, pb, pr = pads
    sH, sW       = strides

    # Effective kernel with dilation
    kH_eff = kH + (kH - 1) * (dH - 1)
    kW_eff = kW + (kW - 1) * (dW - 1)

    # Padded image shape
    H_pad = H_img + pt + pb
    W_pad = W_img + pl + pr

    # Recover channels
    C = Ck // (kH * kW)

    # Output sliding grid size from ONNX spec:
    H_out = (H_pad - kH_eff) // sH + 1
    W_out = (W_pad - kW_eff) // sW + 1
    # Optional sanity check:
    # tf.debugging.assert_equal(L, H_out * W_out)

    # -----------------------------
    # Reshape ONNX (N,C*kH*kW,L) → (N,C,kH,kW,H_out,W_out)
    # -----------------------------
    cols = tf.reshape(input_cols, (N, C, kH, kW, L))
    cols = tf.reshape(cols,       (N, C, kH, kW, H_out, W_out))

    # -----------------------------
    # Prepare padded accumulation buffer
    # -----------------------------
    out_pad = tf.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    # -----------------------------
    # Precompute sliding indices (int32)
    # -----------------------------
    h_idx_base = tf.range(H_out, dtype=tf.int32)[None, :, None]  # (1,H_out,1)
    w_idx_base = tf.range(W_out, dtype=tf.int32)[None, None, :]  # (1,1,W_out)

    # -----------------------------
    # Loop over kernel coords (kH,kW)
    # -----------------------------
    for kh in range(kH):
        for kw in range(kW):

            # Offset inside the effective kernel with dilation
            h_off = kh * dH
            w_off = kw * dW

            # Compute target pixel coordinates contributed by this kernel element
            H_idx = h_off + h_idx_base * sH     # (1,H_out,1), int32
            W_idx = w_off + w_idx_base * sW     # (1,1,W_out), int32

            # Extract the patch slice (N,C,H_out,W_out)
            patch_vals = cols[:, :, kh, kw]
            pv_flat = tf.reshape(patch_vals, [-1])

            # -----------------------------
            # Build flat indices (N*C*H_out*W_out, 4): [n, c, h, w]
            # -----------------------------
            # N dimension
            NN = tf.repeat(tf.range(N, dtype=tf.int32),
                           C * H_out * W_out)[:, None]

            # C dimension
            CC = tf.tile(
                tf.repeat(tf.range(C, dtype=tf.int32),
                          H_out * W_out)[None],
                [N, 1]
            )
            CC = tf.reshape(CC, [-1, 1])

            # H dimension
            HH_base = tf.repeat(tf.reshape(H_idx, [-1]), W_out)
            HH = tf.tile(HH_base[None], [N * C, 1])
            HH = tf.reshape(HH, [-1, 1])
            HH = tf.cast(HH, tf.int32)

            # W dimension
            WW_base = tf.reshape(W_idx, [-1])
            WW = tf.tile(WW_base, [N * C * H_out])
            WW = tf.reshape(WW, [-1, 1])
            WW = tf.cast(WW, tf.int32)

            idx = tf.concat([NN, CC, HH, WW], axis=1)  # (?,4), int32

            # -----------------------------
            # Scatter-add (SUM contributions)
            # -----------------------------
            out_pad = tf.tensor_scatter_nd_add(out_pad, idx, pv_flat)

    # -----------------------------
    # Remove padding, keep NCHW
    # -----------------------------
    out_nchw = out_pad[:, :, pt:pt+H_img, pl:pl+W_img]
    return out_nchw



def demo():
    import torch
    import torch.nn as nn

    # 6-channel, 4x4 image: values 1..(6*4*4)
    img = torch.arange(1, 1 + 6 * 4 * 4).float().reshape(1, 6, 4, 4)  # (N=1,C=6,H=4,W=4)

    # Unfold with 2x2 kernel, stride 2
    unfold = nn.Unfold(kernel_size=2, stride=2)
    cols = unfold(img)

    out_tf = col2im_onnx(
        input_cols=tf.constant(cols.numpy(),dtype=tf.float32),
        image_shape=(4,4),
        block_shape=(2,2),
        dilations=(1,1),
        pads=(0,0,0,0),
        strides=(2,2)
    )
    torch_out = nn.Fold((4,4),2,stride=2)(cols)
    print("Torch fold:\n", torch_out[0,3].numpy())
    print("TF col2im:\n", out_tf.numpy()[0,3])
    print(f'max difference = {(out_tf.numpy() - torch_out.numpy()).max()}')


def run_single_case(N, C, H, W,
                    kernel_size, stride, padding, dilation):

    print(f"\n=== Case: N={N}, C={C}, H={H}, W={W}, "
          f"k={kernel_size}, s={stride}, p={padding}, d={dilation} ===")

    # ----- PyTorch path -----
    img = torch.randn(N, C, H, W, dtype=torch.float32)

    unfold = nn.Unfold(
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )
    cols = unfold(img)  # (N, C*kH*kW, L)

    fold = nn.Fold(
        output_size=(H, W),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride
    )
    torch_out = fold(cols)  # (N,C,H,W)

    # ----- TensorFlow path -----
    input_cols_tf = tf.constant(cols.numpy(), dtype=tf.float32)

    out_tf_nchw = col2im_onnx(
        input_cols=input_cols_tf,
        image_shape=(H, W),
        block_shape=(kernel_size, kernel_size),
        dilations=(dilation, dilation),
        pads=(padding, padding, padding, padding),
        strides=(stride, stride)
    )

    out_tf_np = out_tf_nchw.numpy()          # (N,C,H,W)
    out_torch_np = torch_out.detach().numpy()

    # ----- Compare -----
    max_diff = np.max(np.abs(out_tf_np - out_torch_np))
    print("Torch out shape:", out_torch_np.shape)
    print("TF out shape   :", out_tf_np.shape)
    print("max |TF - Torch| =", max_diff)

    # Optionally inspect first channel of first batch
    # print("Torch[0,0]:\n", out_torch_np[0, 0])
    # print("TF[0,0]:\n", out_tf_np[0, 0])

    return max_diff


def demo_large():
    np.random.seed(0)
    torch.manual_seed(0)

    cases = [
        # (N, C, H, W, k, s, p, d)
        (1, 1, 4, 4, 2, 2, 0, 1),   # simple, non-overlapping, no pad
        (1, 6, 4, 4, 2, 2, 0, 1),   # your 6-channel example
        (1, 3, 7, 7, 3, 1, 0, 1),   # overlapping stride=1
        (2, 3, 8, 8, 3, 2, 1, 1),   # batch>1, padding=1
        (1, 4, 9, 9, 3, 2, 2, 2),   # padding + dilation
        (2, 2, 10, 12, 2, 3, 1, 1), # rectangular W != H, stride>1
    ]

    all_diffs = []

    for (N, C, H, W, k, s, p, d) in cases:
        diff = run_single_case(
            N=N, C=C, H=H, W=W,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d
        )
        all_diffs.append(diff)

    print("\n=== Summary over all cases ===")
    for (cfg, d) in zip(cases, all_diffs):
        N, C, H, W, k, s, p, di = cfg
        print(f"Case N={N},C={C},H={H},W={W},k={k},s={s},p={p},d={di} "
              f"=> max diff {d}")


if __name__ == "__main__":
    demo_large()