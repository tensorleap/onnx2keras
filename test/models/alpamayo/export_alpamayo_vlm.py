"""Export the Alpamayo-R1-10B VLM backbone (Qwen3-VL-8B based) to ONNX with KV-cache I/O.

Three graphs that together cover every VLM weight (all real checkpoint tensors, fp32):

1. vision encoder  (vlm.model.visual.*, ~0.6B params)
     pixel_patches (768, 1536)  [one 512x384 frame, grid 1x24x32, temporal patch 2]
       -> visual_embeds (192, 4096), deepstack_embeds_{0,1,2} (192, 4096)

2. decoder step    (vlm.model.language_model.layers/norm + vlm.lm_head, ~7.6B params)
     inputs_embeds     (1, 1, 4096)
     position_ids      (3, 1, 1)  int64 mrope positions
     attention_mask    (1, 1, 1, P+1) additive float mask (zeros = attend)
     deepstack_embed_{0,1,2} (1, 1, 4096)  visual deepstack injection for the first
                        3 decoder layers (pass zeros for text tokens)
     past_key_values.{i}.key / .value (1, 8, P, 128) x 36, P dynamic
       -> logits (1, 1, 155697)
          present.{i}.key / .value (1, 8, P+1, 128) x 36

3. embed_tokens    (vlm.model.language_model.embed_tokens, 155697 x 4096)
     input_ids (1, 8) int64 -> inputs_embeds (1, 8, 4096)

Generation loop = embed graph (or vision graph for image patches) -> decoder step
per token, feeding present.* back into past_key_values.*. The expert denoiser
(export_alpamayo_full.py) consumes the same cache tensors.

Usage:
    python export_alpamayo_vlm.py --shard-dir <all 5 Alpamayo shards> \
        --out-dir test/models/alpamayo [--only vision|decoder|embed]
"""
import argparse
import gc
import json
import os

import numpy as np
import torch

QWEN3_VL_CONFIG_URL = (
    "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/main/config.json"
)
SHARDS = [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)]

VOCAB_SIZE = 155697  # Alpamayo config.json vocab_size (extended Qwen3 vocab)
HIDDEN = 4096
N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
PAST_LEN = 32  # trace-time past length; the axis is exported as dynamic

# one 512x384 image: grid_thw = (1, 24, 32) -> 768 patches -> 192 merged tokens
GRID_THW = (1, 24, 32)
N_PATCHES = GRID_THW[0] * GRID_THW[1] * GRID_THW[2]
PATCH_DIM = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size^2
N_MERGED = N_PATCHES // 4  # spatial_merge_size^2


def fetch_qwen_config(cache_path):
    if not os.path.exists(cache_path):
        import requests

        r = requests.get(QWEN3_VL_CONFIG_URL, timeout=60)
        r.raise_for_status()
        with open(cache_path, "w") as f:
            f.write(r.text)
    return json.load(open(cache_path))


def load_prefixed_weights(shard_dir, prefix, strip_prefix):
    from safetensors import safe_open

    sd = {}
    for shard in SHARDS:
        path = os.path.join(shard_dir, shard)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    sd[key[len(strip_prefix):]] = f.get_tensor(key).to(torch.float32)
    assert sd, f"no tensors with prefix {prefix} under {shard_dir}"
    return sd


def consolidate_external_data(out_path):
    """Merge the exporter's per-tensor external files into one <name>.onnx.data."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    out_dir = os.path.dirname(os.path.abspath(out_path))
    base = os.path.basename(out_path)
    m = onnx.load(out_path)
    for f in os.listdir(out_dir):
        full = os.path.join(out_dir, f)
        if (os.path.isfile(full) and f != base
                and not f.endswith((".onnx", ".data", ".py", ".json"))):
            os.remove(full)
    convert_model_to_external_data(
        m, all_tensors_to_one_file=True, location=base + ".data", size_threshold=1024
    )
    onnx.save_model(m, out_path)
    del m
    gc.collect()


def ort_check(out_path, feeds, refs, mean_tol=1e-4, max_tol=1e-3):
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path)
    outs = sess.run(None, feeds)
    for i, (o, r) in enumerate(zip(outs, refs)):
        diff = np.abs(o - r.numpy())
        print(f"  out{i}: mean={diff.mean():.3e} max={diff.max():.3e}")
        assert diff.mean() < mean_tol and diff.max() < max_tol, f"output {i} mismatch"
    del sess
    gc.collect()


# --------------------------------- vision ---------------------------------

class VisionWrapper(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        self.register_buffer(
            "grid_thw", torch.tensor([GRID_THW], dtype=torch.long), persistent=False
        )

    def forward(self, pixel_patches):
        embeds, deepstack = self.visual(pixel_patches, grid_thw=self.grid_thw)
        return (embeds, *deepstack)


def export_vision(args, qcfg):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(**qcfg["vision_config"])
    cfg._attn_implementation = "eager"
    visual = Qwen3VLVisionModel(cfg).float().eval()
    sd = load_prefixed_weights(args.shard_dir, "vlm.model.visual.", "vlm.model.visual.")
    visual.load_state_dict(sd, strict=True)
    del sd
    gc.collect()

    wrapper = VisionWrapper(visual).eval()
    patches = torch.rand(N_PATCHES, PATCH_DIM)
    with torch.no_grad():
        refs = wrapper(patches)
    print("vision outputs:", [tuple(r.shape) for r in refs])

    out = os.path.join(args.out_dir, "alpamayo_r1_vision_encoder.onnx")
    torch.onnx.export(
        wrapper, (patches,), out,
        input_names=["pixel_patches"],
        output_names=["visual_embeds", "deepstack_embeds_0",
                      "deepstack_embeds_1", "deepstack_embeds_2"],
        opset_version=17, do_constant_folding=True,
    )
    del wrapper, visual
    gc.collect()
    consolidate_external_data(out)
    ort_check(out, {"pixel_patches": patches.numpy()}, refs)
    print("vision encoder OK ->", out)


# --------------------------------- decoder ---------------------------------

class SplitLMHead(torch.nn.Module):
    """lm_head with the 155697x4096 fp32 weight (2.55GB) split along vocab into
    chunks < 2GB each (protobuf cannot serialize a single tensor above 2GiB),
    logits = concat of the chunk projections - mathematically identical."""

    N_CHUNKS = 4

    def __init__(self, weight):
        super().__init__()
        chunks = torch.chunk(weight, self.N_CHUNKS, dim=0)
        self.heads = torch.nn.ModuleList()
        for c in chunks:
            lin = torch.nn.Linear(c.shape[1], c.shape[0], bias=False)
            lin.weight.data.copy_(c)
            self.heads.append(lin)

    def forward(self, hidden):
        return torch.cat([h(hidden) for h in self.heads], dim=-1)


class SplitEmbed(torch.nn.Module):
    """embed_tokens with the weight split along the hidden dim (same 2GiB
    single-tensor protobuf limit), output = concat of the partial lookups."""

    N_CHUNKS = 2

    def __init__(self, weight):
        super().__init__()
        chunks = torch.chunk(weight, self.N_CHUNKS, dim=1)
        self.embeds = torch.nn.ModuleList(
            [torch.nn.Embedding(c.shape[0], c.shape[1], _weight=c.contiguous())
             for c in chunks]
        )

    def forward(self, input_ids):
        return torch.cat([e(input_ids) for e in self.embeds], dim=-1)

class DecodeStepWrapper(torch.nn.Module):
    """One text-generation step with explicit KV-cache tensors.

    Reimplements Qwen3VLTextModel.forward's loop with the same submodules so that
    (a) the DynamicCache is built from / returned as plain tensors and
    (b) the deepstack visual injection becomes a plain add of full-length inputs
        (zeros outside image positions) instead of an untraceable masked scatter.
    """

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds, position_ids, attention_mask, ds0, ds1, ds2,
                *past_flat):
        from transformers.cache_utils import DynamicCache

        deepstack = [ds0, ds1, ds2]
        cache = DynamicCache(config=self.model.config)
        for i in range(N_LAYERS):
            cache.update(past_flat[2 * i], past_flat[2 * i + 1], i)

        hidden = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden, position_ids)
        text_position_ids = position_ids[0]

        for i, layer in enumerate(self.model.layers):
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            if i < len(deepstack):
                hidden = hidden + deepstack[i]

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        presents = []
        for i in range(N_LAYERS):
            presents.extend([cache.layers[i].keys, cache.layers[i].values])
        return (logits, *presents)


def export_decoder(args, qcfg):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(**qcfg["text_config"])
    cfg.vocab_size = VOCAB_SIZE
    cfg._attn_implementation = "eager"
    text_model = Qwen3VLTextModel(cfg).float().eval()

    print("loading language model weights (~30GB fp32)...")
    sd = load_prefixed_weights(
        args.shard_dir, "vlm.model.language_model.", "vlm.model.language_model."
    )
    text_model.load_state_dict(sd, strict=True)
    del sd
    gc.collect()

    head_sd = load_prefixed_weights(args.shard_dir, "vlm.lm_head.", "vlm.lm_head.")
    lm_head = SplitLMHead(head_sd["weight"])
    del head_sd
    gc.collect()

    # embed_tokens is exported as its own graph; drop it from this one
    embed_weight = text_model.embed_tokens.weight.detach().clone()
    del text_model.embed_tokens
    gc.collect()

    wrapper = DecodeStepWrapper(text_model, lm_head).eval()

    torch.manual_seed(0)
    x = torch.rand(1, 1, HIDDEN) * 0.02
    pos = torch.full((3, 1, 1), PAST_LEN, dtype=torch.long)
    mask = torch.zeros(1, 1, 1, PAST_LEN + 1)
    ds = [torch.zeros(1, 1, HIDDEN) for _ in range(3)]
    past = [torch.randn(1, N_KV_HEADS, PAST_LEN, HEAD_DIM) * 0.1
            for _ in range(2 * N_LAYERS)]
    inputs = (x, pos, mask, *ds, *past)
    with torch.no_grad():
        refs = wrapper(*inputs)
    print("decoder logits:", tuple(refs[0].shape), "present0:", tuple(refs[1].shape))

    past_names = []
    for i in range(N_LAYERS):
        past_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
    present_names = []
    for i in range(N_LAYERS):
        present_names += [f"present.{i}.key", f"present.{i}.value"]
    dynamic_axes = {n: {2: "past_len"} for n in past_names}
    dynamic_axes.update({n: {2: "past_len_plus_1"} for n in present_names})
    dynamic_axes["attention_mask"] = {3: "past_len_plus_1"}

    out = os.path.join(args.out_dir, "alpamayo_r1_vlm_decoder.onnx")
    print("exporting decoder ONNX (~30GB external data)...")
    torch.onnx.export(
        wrapper, inputs, out,
        input_names=["inputs_embeds", "position_ids", "attention_mask",
                     "deepstack_embed_0", "deepstack_embed_1", "deepstack_embed_2",
                     *past_names],
        output_names=["logits", *present_names],
        dynamic_axes=dynamic_axes,
        opset_version=17, do_constant_folding=True,
    )
    del wrapper, text_model, lm_head
    gc.collect()
    consolidate_external_data(out)

    feeds = {"inputs_embeds": x.numpy(), "position_ids": pos.numpy(),
             "attention_mask": mask.numpy(),
             **{f"deepstack_embed_{i}": ds[i].numpy() for i in range(3)},
             **{n: p.numpy() for n, p in zip(past_names, past)}}
    # logits tolerance is looser: 36 fp32 layers + a 155k-dim matmul
    ort_check(out, feeds, refs, mean_tol=1e-3, max_tol=1e-2)

    # prove the past axis is truly dynamic: rerun with P=3
    import onnxruntime as ort
    sess = ort.InferenceSession(out)
    small = {**feeds,
             "attention_mask": np.zeros((1, 1, 1, 4), np.float32),
             **{n: p.numpy()[:, :, :3] for n, p in zip(past_names, past)}}
    small["position_ids"] = np.full((3, 1, 1), 3, np.int64)
    outs = sess.run(["logits", "present.0.key"], small)
    assert outs[1].shape == (1, N_KV_HEADS, 4, HEAD_DIM)
    del sess
    gc.collect()
    print("decoder (dynamic past OK) ->", out)
    return embed_weight


def export_embed(args, embed_weight):
    # torch.onnx.export's shape-inference pass trips on the 2GiB protobuf limit
    # even with the weight chunked, so build this trivial graph (2 Gathers +
    # Concat over hidden-dim halves of embed_tokens) directly with onnx.helper.
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import convert_model_to_external_data

    embed = SplitEmbed(embed_weight).eval()
    ids = torch.arange(8).view(1, 8)
    with torch.no_grad():
        ref = embed(ids)

    chunks = torch.chunk(embed_weight, SplitEmbed.N_CHUNKS, dim=1)
    inits, gathers, part_names = [], [], []
    for i, c in enumerate(chunks):
        w_name, o_name = f"embed_tokens_part{i}", f"embeds_part{i}"
        inits.append(numpy_helper.from_array(c.contiguous().numpy(), w_name))
        gathers.append(helper.make_node("Gather", [w_name, "input_ids"], [o_name],
                                        axis=0, name=f"gather_{i}"))
        part_names.append(o_name)
    concat = helper.make_node("Concat", part_names, ["inputs_embeds"], axis=-1,
                              name="concat_embeds")
    graph = helper.make_graph(
        [*gathers, concat], "alpamayo_r1_vlm_embed",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT,
                                       [1, 8, HIDDEN])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    out = os.path.join(args.out_dir, "alpamayo_r1_vlm_embed.onnx")
    convert_model_to_external_data(
        model, all_tensors_to_one_file=True,
        location=os.path.basename(out) + ".data", size_threshold=1024,
    )
    onnx.save_model(model, out)
    del model
    gc.collect()
    ort_check(out, {"input_ids": ids.numpy()}, [ref])
    print("embed OK ->", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--only", choices=["vision", "decoder", "embed"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qcfg = fetch_qwen_config(os.path.join(args.shard_dir, "qwen3vl_config.json"))

    if args.only in (None, "vision"):
        export_vision(args, qcfg)
    if args.only in (None, "decoder", "embed"):
        embed_weight = None
        if args.only != "embed":
            embed_weight = export_decoder(args, qcfg)
        if args.only in (None, "embed"):
            if embed_weight is None:
                sd = load_prefixed_weights(
                    args.shard_dir,
                    "vlm.model.language_model.embed_tokens.",
                    "vlm.model.language_model.embed_tokens.",
                )
                embed_weight = sd["weight"]
            export_embed(args, embed_weight)
    print("ALL DONE")


if __name__ == "__main__":
    main()
