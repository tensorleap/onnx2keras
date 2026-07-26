"""Export the Alpamayo-R1-10B diffusion-denoiser step (full 2.3B action expert) to ONNX.

This reproduces the NEURAL part of AlpamayoR1's denoising `step_fn`
(see NVlabs/alpamayo src/alpamayo_r1/models/alpamayo_r1.py):

    action_in_proj (PerWaypointActionInProjV2)
      -> expert (36-layer Qwen3-VL text decoder, hidden 2048, 16Q/8KV heads, FFN 8256)
      -> action_out_proj (Linear 2048 -> 2)

    inputs : noisy_action (1, 64, 2) f32, timesteps (1, 1, 1) f32
    output : vector_field (1, 64, 2) f32

Differences vs. real inference (documented, intentional - the ONNX must be
self-contained with only the two diffusion inputs):
  * no VLM prompt KV-cache: the expert runs on the 64 action tokens only,
    with full (non-causal) self-attention, as expert_non_causal_attention=True.
  * position_ids are the constant arange(64) mrope positions (3, B, 64);
    at inference they are shifted by the prompt length, which only offsets RoPE phases.
  * weights are the real bf16 checkpoint tensors upcast to fp32.

Requirements (separate env from onnx2kerastl): torch>=2.5, transformers==4.57.1,
onnx, onnxruntime, safetensors, requests, numpy. Weight shards 4 and 5 of
nvidia/Alpamayo-R1-10B must be downloaded first (see --shard-dir).

Usage:
    python export_alpamayo_full.py --shard-dir /path/to/shards \
        --out test/models/alpamayo/alpamayo_r1_full_denoiser.onnx
"""
import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import torch

QWEN3_VL_CONFIG_URL = (
    "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/main/config.json"
)
SHARDS = ["model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors"]

# config.json (nvidia/Alpamayo-R1-10B): expert_cfg overrides applied on top of the
# Qwen3-VL-8B text_config, mirroring AlpamayoR1.__init__.
EXPERT_CFG = {
    "head_dim": 128,
    "hidden_size": 2048,
    "intermediate_size": 8256,
    "num_attention_heads": 16,
}
N_WAYPOINTS = 64
ACTION_DIM = 2
EXPERT_HIDDEN = 2048


def get_text_config(cache_path):
    """Qwen3-VL-8B text config with expert overrides, as AlpamayoR1 builds it."""
    from transformers.models.qwen3_vl import Qwen3VLConfig

    if not os.path.exists(cache_path):
        import requests

        r = requests.get(QWEN3_VL_CONFIG_URL, timeout=60)
        r.raise_for_status()
        with open(cache_path, "w") as f:
            f.write(r.text)
    vlm_config = Qwen3VLConfig(**json.load(open(cache_path)))
    expert_config = copy.deepcopy(vlm_config.text_config)
    for key, value in EXPERT_CFG.items():
        setattr(expert_config, key, value)
    expert_config._attn_implementation = "eager"
    return expert_config


def load_action_in_proj_module(alpamayo_src):
    spec = importlib.util.spec_from_file_location(
        "action_in_proj", f"{alpamayo_src}/src/alpamayo_r1/models/action_in_proj.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AlpamayoDenoiser(torch.nn.Module):
    """Self-contained step_fn: noisy action + timestep -> predicted vector field."""

    def __init__(self, action_in_proj, expert, action_out_proj):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        # mrope positions (3, B, T): text-only rollout uses the same arange on all
        # three rows; the prompt-length shift at inference only offsets RoPE phases.
        self.register_buffer(
            "position_ids",
            torch.arange(N_WAYPOINTS).view(1, 1, -1).expand(3, 1, -1).contiguous(),
            persistent=False,
        )
        # expert_non_causal_attention=True -> all 64 action tokens fully visible:
        # an explicit all-zero float mask (B, 1, T, T) sidesteps causal-mask creation.
        self.register_buffer(
            "attention_mask",
            torch.zeros(1, 1, N_WAYPOINTS, N_WAYPOINTS),
            persistent=False,
        )

    def forward(self, x, t):
        future_token_embeds = self.action_in_proj(x, t)  # (B, 64, 2048)
        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=self.position_ids,
            attention_mask=self.attention_mask,
            use_cache=False,
        )
        last_hidden = expert_out.last_hidden_state[:, -N_WAYPOINTS:]
        return self.action_out_proj(last_hidden).view(-1, N_WAYPOINTS, ACTION_DIM)


def load_weights(shard_dir):
    from safetensors import safe_open

    sd = {}
    for shard in SHARDS:
        path = os.path.join(shard_dir, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(("expert.", "action_in_proj.", "action_out_proj.")):
                    sd[key] = f.get_tensor(key).to(torch.float32)
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-dir", required=True,
                        help="dir holding Alpamayo-R1-10B shards 4 and 5")
    parser.add_argument("--alpamayo-src", required=True,
                        help="path to a clone of https://github.com/NVlabs/alpamayo")
    args = parser.parse_args()

    torch.manual_seed(0)

    print("Building modules...")
    from transformers import AutoModel

    text_config = get_text_config(os.path.join(args.shard_dir, "qwen3vl_config.json"))
    expert = AutoModel.from_config(text_config)
    del expert.embed_tokens  # AlpamayoR1 deletes it; checkpoint has no expert.embed_tokens
    print("expert:", type(expert).__name__)

    aip_mod = load_action_in_proj_module(args.alpamayo_src)
    action_in_proj = aip_mod.PerWaypointActionInProjV2(
        in_dims=[N_WAYPOINTS, ACTION_DIM],
        out_dim=EXPERT_HIDDEN,
        num_enc_layers=2,
        hidden_size=512,
        max_freq=100.0,
        num_fourier_feats=20,
    )
    action_out_proj = torch.nn.Linear(EXPERT_HIDDEN, ACTION_DIM)

    # Qwen3-VL config carries torch_dtype=bfloat16; the exported graph is fp32
    model = AlpamayoDenoiser(action_in_proj, expert, action_out_proj).float()

    print("Loading checkpoint tensors from shards...")
    sd = load_weights(args.shard_dir)
    print(f"  {len(sd)} tensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "position_ids" not in k and "attention_mask" not in k]
    assert not real_missing, f"missing weights: {real_missing[:5]}"
    assert not unexpected, f"unexpected weights: {unexpected[:5]}"
    model.eval()

    x = torch.rand(1, N_WAYPOINTS, ACTION_DIM)
    t = torch.rand(1, 1, 1)
    with torch.no_grad():
        ref = model(x, t)
    print("torch output:", tuple(ref.shape))

    # non-causality check: with full attention, perturbing the LAST token must
    # change the FIRST token's output.
    x2 = x.clone()
    x2[0, -1] += 0.5
    with torch.no_grad():
        ref2 = model(x2, t)
    first_tok_delta = (ref2[0, 0] - ref[0, 0]).abs().max().item()
    assert first_tok_delta > 0, "expert attention is causal - non-causal path not active"
    print(f"non-causal check ok (first-token delta {first_tok_delta:.2e})")

    print("Exporting ONNX (external data for >2GB)...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    kwargs = dict(
        input_names=["noisy_action", "timesteps"],
        output_names=["vector_field"],
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (x, t), args.out, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (x, t), args.out, **kwargs)
    print("exported", args.out)

    # the classic exporter scatters one external-data file per tensor;
    # consolidate into a single <name>.onnx.data next to the model
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    out_dir = os.path.dirname(os.path.abspath(args.out))
    base = os.path.basename(args.out)
    m = onnx.load(args.out)
    for f in os.listdir(out_dir):
        if f != base and not f.endswith((".onnx", ".data", ".py", ".json")):
            os.remove(os.path.join(out_dir, f))
    convert_model_to_external_data(
        m, all_tensors_to_one_file=True, location=base + ".data", size_threshold=1024
    )
    onnx.save_model(m, args.out)
    del m
    print("consolidated external data ->", base + ".data")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out)
    out = sess.run(None, {"noisy_action": x.numpy(), "timesteps": t.numpy()})[0]
    diff = np.abs(out - ref.numpy())
    print(f"onnxruntime vs torch: mean={diff.mean():.3e} max={diff.max():.3e}")
    assert diff.mean() < 1e-4 and diff.max() < 1e-3
    print("OK")


if __name__ == "__main__":
    main()
