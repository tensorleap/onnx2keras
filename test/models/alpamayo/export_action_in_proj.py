"""Export Alpamayo-R1-10B `action_in_proj` (PerWaypointActionInProjV2) to ONNX.

Downloads ONLY the 13 action_in_proj.* tensors from model-00005-of-00005.safetensors
via HTTP range requests (they are tiny), loads them into the real NVlabs/alpamayo
module code, and exports a fixed-shape ONNX:
    inputs : noisy_action (1, 64, 2) f32, timesteps (1, 1, 1) f32
    output : action_embeds (1, 64, 2048) f32
"""
import importlib.util
import json
import struct
import sys

import numpy as np
import requests
import torch

SHARD_URL = (
    "https://huggingface.co/nvidia/Alpamayo-R1-10B/resolve/main/"
    "model-00005-of-00005.safetensors"
)
ALPAMAYO_SRC = sys.argv[1]  # path to cloned NVlabs/alpamayo repo
OUT_PATH = sys.argv[2]      # output .onnx path

PREFIX = "action_in_proj."


def fetch(url, start, end):  # inclusive byte range
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    assert r.status_code == 206, f"expected partial content, got {r.status_code}"
    return r.content


def bf16_bytes_to_f32(buf):
    u16 = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def load_action_in_proj_state_dict():
    header_len = struct.unpack("<Q", fetch(SHARD_URL, 0, 7))[0]
    header = json.loads(fetch(SHARD_URL, 8, 8 + header_len - 1))
    data_start = 8 + header_len

    sd = {}
    for name, meta in header.items():
        if not name.startswith(PREFIX):
            continue
        begin, end = meta["data_offsets"]
        raw = fetch(SHARD_URL, data_start + begin, data_start + end - 1)
        dtype = meta["dtype"]
        if dtype == "BF16":
            arr = bf16_bytes_to_f32(raw)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        else:
            raise ValueError(f"unexpected dtype {dtype} for {name}")
        arr = arr.reshape(meta["shape"]).copy()
        sd[name[len(PREFIX):]] = torch.from_numpy(arr)
        print(f"  fetched {name}: {meta['shape']} {dtype}")
    return sd


def build_module():
    spec = importlib.util.spec_from_file_location(
        "action_in_proj", f"{ALPAMAYO_SRC}/src/alpamayo_r1/models/action_in_proj.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # args mirror AlpamayoR1.__init__: in_dims = action_space dims (64 waypoints, 2),
    # out_dim = expert hidden_size (2048); rest from config.action_in_proj_cfg
    return mod.PerWaypointActionInProjV2(
        in_dims=[64, 2],
        out_dim=2048,
        num_enc_layers=2,
        hidden_size=512,
        max_freq=100.0,
        num_fourier_feats=20,
    )


def main():
    print("Fetching action_in_proj weights via range requests...")
    sd = load_action_in_proj_state_dict()
    model = build_module()
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    model.eval()

    x = torch.rand(1, 64, 2)
    t = torch.rand(1, 1, 1)
    with torch.no_grad():
        ref = model(x, t)
    print("torch output:", tuple(ref.shape), ref.dtype)

    kwargs = dict(
        input_names=["noisy_action", "timesteps"],
        output_names=["action_embeds"],
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (x, t), OUT_PATH, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (x, t), OUT_PATH, **kwargs)
    print("exported", OUT_PATH)

    import onnxruntime as ort

    sess = ort.InferenceSession(OUT_PATH)
    out = sess.run(None, {"noisy_action": x.numpy(), "timesteps": t.numpy()})[0]
    diff = np.abs(out - ref.numpy())
    print(f"onnxruntime vs torch: mean={diff.mean():.3e} max={diff.max():.3e}")
    assert diff.max() < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
