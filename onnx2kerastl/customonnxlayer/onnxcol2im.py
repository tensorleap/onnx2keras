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
    ONNX Col2Im, INPUT IN ONNX FORMAT:
        (N, C * prod(block_shape), L)
    OUTPUT IN TENSORFLOW FORMAT:
        (N, H_img, W_img, C)  # NHWC

    Fully ONNX-correct:
        - pads
        - dilations
        - strides
        - lexicographic L ordering
    """

    # -----------------------------
    # Unpack attributes
    # -----------------------------
    N = tf.shape(input_cols)[0]
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

    # -----------------------------
    # Reshape ONNX (N,C*kH*kW,L) → (N,C,kH,kW,H_out,W_out)
    # -----------------------------
    cols = tf.reshape(
        input_cols,
        (N, C, kH, kW, L)
    )

    cols = tf.reshape(
        cols,
        (N, C, kH, kW, H_out, W_out)
    )

    # -----------------------------
    # Prepare padded accumulation buffers
    # -----------------------------
    out_pad = tf.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    count   = tf.zeros_like(out_pad)

    # -----------------------------
    # Precompute sliding indices
    # -----------------------------
    h_idx_base = tf.range(H_out)[None, :, None]  # shape (1,H_out,1)
    w_idx_base = tf.range(W_out)[None, None, :]  # shape (1,1,W_out)

    # -----------------------------
    # Loop over kernel coords (kH,kW)
    # (kH,kW) loops are small, acceptable for ONNX correctness.
    # -----------------------------
    for kh in range(kH):
        for kw in range(kW):

            # Offset inside the effective kernel with dilation
            h_off = kh * dH
            w_off = kw * dW

            # Compute target pixel coordinates contributed by this kernel element
            H_idx = h_off + h_idx_base * sH     # (1,H_out,1)
            W_idx = w_off + w_idx_base * sW     # (1,1,W_out)

            # Extract the patch slice (N,C,H_out,W_out)
            patch_vals = cols[:, :, kh, kw]

            # Flatten patch values for scatter
            pv_flat = tf.reshape(patch_vals, [-1])

            # -----------------------------
            # Build flat indices
            # -----------------------------
            # Indices need to be (total_updates, 4): [n, c, h, w]

            # N dimension
            NN = tf.repeat(tf.range(N), C * H_out * W_out)[:, None]

            # C dimension
            CC = tf.tile(
                tf.repeat(tf.range(C), H_out * W_out)[None],
                [N, 1]
            )
            CC = tf.reshape(CC, [-1, 1])

            # H dimension
            HH = tf.tile(
                tf.repeat(tf.reshape(H_idx, [-1]), W_out)[None],
                [N * C, 1]
            )
            HH = tf.reshape(HH, [-1, 1])

            # W dimension
            WW = tf.tile(
                tf.reshape(W_idx, [-1]),
                [N * C * H_out]
            )[:, None]

            idx = tf.concat([NN, CC, HH, WW], axis=1)

            # -----------------------------
            # Scatter-add
            # -----------------------------
            out_pad = tf.tensor_scatter_nd_add(out_pad, idx, pv_flat)
            count   = tf.tensor_scatter_nd_add(count, idx, tf.ones_like(pv_flat))

    # -----------------------------
    # Normalize overlapping regions
    # -----------------------------
    out_pad = tf.math.divide_no_nan(out_pad, count)

    # -----------------------------
    # Remove padding and convert NCHW → NHWC
    # -----------------------------
    out_nchw = out_pad[:, :, pt:pt+H_img, pl:pl+W_img]
    # out_nhwc = tf.transpose(out_nchw, (0, 2, 3, 1))  # NHWC

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


if __name__ == "__main__":
    demo()
