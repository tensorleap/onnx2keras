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


N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
PAST_LEN = 32  # trace-time VLM prompt-cache length; the axis is exported as dynamic


class AlpamayoDenoiser(torch.nn.Module):
    """step_fn with the VLM prompt KV-cache as explicit inputs.

    At inference AlpamayoR1's diffusion loop runs the expert conditioned on the
    VLM's past_key_values (the expert shares the VLM cache layout: 8 KV heads x 128).
    Inputs mirror that call:
      noisy_action (1, 64, 2), timesteps (1, 1, 1)
      position_ids (3, 1, 64)  mrope positions (arange(64) + rope_delta + prompt offset)
      attention_mask (1, 1, 64, P+64) additive float mask (zeros = attend,
          large negative = masked; the real pipeline masks left-padding this way)
      past_key_values.{i}.key/.value (1, 8, P, 128) x 36 - feed the VLM decoder's
          present.* outputs here
    Output: vector_field (1, 64, 2). No present.* outputs: the pipeline crops the
    cache back to the prompt length after every denoising step, so the expert's
    appended entries are never reused.

    The loop reimplements Qwen3VLTextModel.forward with the same submodules so the
    cache is built from plain tensors and the input mask reaches attention directly.
    """

    def __init__(self, action_in_proj, expert, action_out_proj):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj

    def forward(self, x, t, position_ids, attention_mask, *past_flat):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.expert.config)
        for i in range(N_LAYERS):
            cache.update(past_flat[2 * i], past_flat[2 * i + 1], i)

        hidden = self.action_in_proj(x, t)  # (B, 64, 2048)
        position_embeddings = self.expert.rotary_emb(hidden, position_ids)
        text_position_ids = position_ids[0]
        for layer in self.expert.layers:
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        hidden = self.expert.norm(hidden)
        last_hidden = hidden[:, -N_WAYPOINTS:]
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

    torch.manual_seed(0)
    x = torch.rand(1, N_WAYPOINTS, ACTION_DIM)
    t = torch.rand(1, 1, 1)
    pos = (torch.arange(N_WAYPOINTS) + PAST_LEN).view(1, 1, -1).expand(3, 1, -1).contiguous()
    mask = torch.zeros(1, 1, N_WAYPOINTS, PAST_LEN + N_WAYPOINTS)
    past = [torch.randn(1, N_KV_HEADS, PAST_LEN, HEAD_DIM) * 0.1
            for _ in range(2 * N_LAYERS)]
    inputs = (x, t, pos, mask, *past)
    with torch.no_grad():
        ref = model(*inputs)
    print("torch output:", tuple(ref.shape))

    # parity check: the reimplemented loop must match the official
    # Qwen3VLTextModel.forward with an equivalent prefilled DynamicCache
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.expert.config)
    for i in range(N_LAYERS):
        cache.update(past[2 * i], past[2 * i + 1], i)
    with torch.no_grad():
        embeds = model.action_in_proj(x, t)
        official = model.expert(
            inputs_embeds=embeds, position_ids=pos, attention_mask=mask,
            past_key_values=cache, use_cache=True,
        ).last_hidden_state
        official = model.action_out_proj(official[:, -N_WAYPOINTS:]).view(
            -1, N_WAYPOINTS, ACTION_DIM)
    parity = (official - ref).abs().max().item()
    assert parity < 1e-5, f"wrapper diverges from official forward: {parity}"
    print(f"official-forward parity ok (max delta {parity:.2e})")

    # non-causality check: with a zero mask, perturbing the LAST action token
    # must change the FIRST token's output.
    x2 = x.clone()
    x2[0, -1] += 0.5
    with torch.no_grad():
        ref2 = model(x2, t, pos, mask, *past)
    first_tok_delta = (ref2[0, 0] - ref[0, 0]).abs().max().item()
    assert first_tok_delta > 0, "expert attention is causal - non-causal path not active"
    print(f"non-causal check ok (first-token delta {first_tok_delta:.2e})")

    print("Exporting ONNX (external data for >2GB)...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    past_names = []
    for i in range(N_LAYERS):
        past_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    dynamic_axes = {n: {2: "past_len"} for n in past_names}
    dynamic_axes["attention_mask"] = {3: "past_len_plus_64"}
    torch.onnx.export(
        model, inputs, args.out,
        input_names=["noisy_action", "timesteps", "position_ids",
                     "attention_mask", *past_names],
        output_names=["vector_field"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
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
    feeds = {"noisy_action": x.numpy(), "timesteps": t.numpy(),
             "position_ids": pos.numpy(), "attention_mask": mask.numpy(),
             **{n: p.numpy() for n, p in zip(past_names, past)}}
    out = sess.run(None, feeds)[0]
    diff = np.abs(out - ref.numpy())
    print(f"onnxruntime vs torch: mean={diff.mean():.3e} max={diff.max():.3e}")
    assert diff.mean() < 1e-4 and diff.max() < 1e-3

    # prove the past axis is truly dynamic: rerun with P=3
    small = {**feeds,
             "attention_mask": np.zeros((1, 1, N_WAYPOINTS, 3 + N_WAYPOINTS),
                                        np.float32),
             **{n: p.numpy()[:, :, :3] for n, p in zip(past_names, past)}}
    assert sess.run(None, small)[0].shape == (1, N_WAYPOINTS, ACTION_DIM)
    print("dynamic past OK")
    print("OK")


if __name__ == "__main__":
    main()
