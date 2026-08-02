# ONNX → Keras conversion overhead: findings and fix plan

Investigation of whether `onnx2kerastl` produces "overblown" Keras models that make
inference slow. Triggered by `overwatch/model/v7_5_raw.onnx`, then generalised to the
13 ONNX models available locally in this repo.

**Status:** findings validated, fix plan approved, implementation not started.

---

## 1. TL;DR

There are **two independent defects**, both general (not specific to any one model):

| # | Defect | Who it hits | Cost |
|---|---|---|---|
| **A** | **Transpose sandwich** — every `Conv`/`Pooling` layer is wrapped in its own `Permute`-in/`Permute`-out pair, so the tensor ping-pongs NHWC↔NCHW around every convolution | every CNN, without exception (~2.0–2.6 Permutes per conv) | **~30% of all compute**; 2.5× slower than onnxruntime on `v7_5_raw`, up to **40.6×** on `x3d_s` |
| **B** | **Lambda explosion + eager dispatch** — transformer graphs expand into thousands of `TFOpLambda`/`Lambda` layers, and running them eagerly costs pure Python dispatch | transformers (`swin` 3992 lambdas, `rtdetrv2` 1225, `traffic_light` 559, `dinov2` 407) | eager-vs-graph penalty of **2×–37×**, zero of it real compute |

For the originating question — **`v7_5_raw` is defect A, not B.** Its 57 lambda layers are
all scalar shape plumbing (`Shape`/`Gather`/`Unsqueeze` on `()`- and `(4,)`-shaped tensors)
and cost essentially nothing. The 136 `Permute` layers are the problem.

Conversion is **numerically correct** in all cases checked — max abs diff vs onnxruntime
on `v7_5_raw` is `2.3e-05` across all three outputs.

---

## 2. Environment

All measurements on one machine, single process, CPU only.

```
repo            onnx2keras @ master, commit b54841e ("Fix gather elements take along axis (#229)")
onnx2kerastl    0.0.197
keras-data-format-converter  0.1.24
python          3.9.25
tensorflow      2.12.0
keras           2.12.0
onnx            1.13.0
onnxruntime     1.17.3   (CPUExecutionProvider)
numpy           1.23.5
platform        macOS-15.7.3 arm64 (Apple silicon), 16 logical CPUs
```

**Variance caveat.** These are laptop measurements. Repeated runs of the same
configuration varied by roughly ±10–20% (e.g. `v7_5_raw` eager was measured at 92.4,
93.3, 94.0, 94.7, 98.2 and 119.0 ms across six runs). Treat single numbers as indicative
and ratios as the meaningful quantity. Every ratio below compares onnxruntime and Keras
on **the same input tensor in the same process**, so the comparison itself is fair even
where absolute times drift.

---

## 3. Methodology

### 3.1 Conversion pipeline under test

The pipeline is exactly what `onnx2kerastl/convert_model.py:8-26` does in production:

```python
onnx_model = onnx.load(path)
input_features = [inp.name for inp in onnx_model.graph.input]   # minus initializers
keras_model = onnx_to_keras(onnx_model, input_names=input_features,
                            name_policy='attach_weights_name',
                            allow_partial_compilation=False).converted_model
final_model = convert_channels_first_to_last(keras_model,
                                             should_transform_inputs_and_outputs=False)
```

The second step matters: `onnx_to_keras` alone emits a **channels-first** model, which
cannot run on CPU at all (`Conv2D op currently only supports the NHWC tensor format on the
CPU`). So every measurement below is of the post-`convert_channels_first_to_last` model,
i.e. what actually ships.

### 3.2 Timing protocol

```python
def bench(fn, arg, n, warm=3):
    for _ in range(warm): fn(arg)          # warm-up excluded (covers tracing/JIT)
    t0 = time.perf_counter()
    for _ in range(n): fn(arg)
    return (time.perf_counter() - t0) / n
```

Four execution modes were timed, because they differ substantially:

| mode | how | what it measures |
|---|---|---|
| `onnxruntime` | `sess.run(None, feed)` | baseline |
| `keras eager` | `model(x)` on a `tf.constant` | Python walks every layer, dispatching ops one at a time |
| `keras graph` | `tf.function(lambda t: model(t)).get_concrete_function(spec)` | the compiled, Grappler-optimised graph — the fastest the h5 can go |
| `keras predict` | `model.predict(x, verbose=0)` | graph-compiled, plus data-adapter / batching / callback plumbing |

### 3.3 Layer and op counting

- **Keras layer histogram**: `collections.Counter(type(l).__name__ for l in model.layers)`.
- **Graph op histogram**: `Counter(o.type for o in concrete_fn.graph.get_operations())` —
  this is the *post-Grappler* graph, so it proves whether TF's own optimiser folds the
  transposes away. It does not.
- **Frozen graph check**: `convert_variables_to_constants_v2(concrete_fn)` then the same
  counter, to rule out variable-reading noise.

### 3.4 Per-op cost attribution (the 30% figure)

Two independent methods, which agree:

**Method 1 — standalone re-timing.** For each `Permute` layer, recover its true input
shape by running a sub-model up to that layer, allocate a real `tf.Variable` of that
shape, and time `tf.function(lambda v: tf.transpose(v, perm))` in isolation:

```python
sub  = tf.keras.Model(final.input, [l.output for l in perm_layers])
outs = sub(x)                                    # concrete shapes at 640x640
for o, l in zip(outs, perm_layers):
    inshape = <outshape un-permuted by l.dims>
    src = tf.Variable(tf.random.uniform(inshape))
    f = tf.function(lambda v: tf.transpose(v, (0,)+tuple(l.dims)))
    ... time f(src) over 20 iterations
```

Result for `v7_5_raw`: **26.1 ms** across 136 Permutes, against a 94.7 ms full-model eager
time → ≈28%.

> An earlier attempt that timed `tf.transpose(tf.zeros(shape), perm)` returned 0.4 ms and
> was **discarded** — TF constant-folds `transpose(zeros)` at trace time, so it measured
> nothing. Using real `tf.Variable` inputs defeats the folding.

**Method 2 — TensorFlow profiler (authoritative).** Profile 10 iterations of the
graph-mode concrete function, then parse the emitted `*.xplane.pb` protobuf directly
(`tensorflow.core.profiler.protobuf.xplane_pb2`), aggregating leaf events on the
`tf_Compute/*` lines by the op-type suffix of the event name:

```python
for plane in xspace.planes:
    ev = {e.id: e.name for e in plane.event_metadata.values()}
    for line in plane.lines:
        if not line.name.startswith('tf_Compute'): continue
        for e in line.events:
            name = ev[e.metadata_id]           # e.g. "model_2/permute_50/transpose:Transpose"
            optype = name.rsplit(':', 1)[1]
            agg[optype] += e.duration_ps
```

> The `tf_op` XStat is not populated in TF 2.12, hence keying off the event-name suffix.
> The generic `.trace.json.gz` path does not exist in this TF version either — only
> `xplane.pb` is written, so it must be parsed directly.
>
> These durations are **CPU time summed across worker threads**, not wall time (238 ms of
> thread-time against ~51 ms wall). They are therefore valid as *relative shares*, which
> is how they are used below, and not as absolute latencies. The `ExecutorState::Process`
> scope (126.88 ms, 139 events) is the executor wrapper enclosing the real ops and is
> excluded from the share denominator to avoid double-counting.

### 3.5 Input shapes for the survey

Input tensors were synthesised per model from the ONNX input spec, with dynamic dims
resolved by name: `batch`→1, `height`→640, `width`→640, and any other unrecognised
symbolic dim → 224 for rank-4 inputs, else 1.

**Two models were mis-resolved by that fallback and their absolute times are not
representative:**

- `swin_v2_t` — input is `['batch_size', 3, 224, 224]`; `batch_size` did not match the
  table, so it ran at **batch 224**. Hence the 4294 ms onnxruntime figure.
- `chip` seg — input is `['unk__251', 250, 220, 1]`; ran at **first-dim 224**. Hence
  693 ms.

Both backends received the identical tensor, so the **ratios for these two rows remain
valid**; only the absolute milliseconds are meaningless. `test/model.onnx` (a BERT-like
model) ran at sequence length 1, which is trivially small — its absolute times should
likewise be ignored.

---

## 4. Deep dive: `v7_5_raw.onnx`

### 4.1 Source model

```
opset      12
input      images  [1, height, width, 3]        (NHWC, dynamic H/W)
outputs    p3 [1,3,ny,nx,14]   p4 [1,3,ny,nx,14]   p5 [1,3,ny,nx,14]
nodes      191
op mix     Conv 58, LeakyRelu 55, Constant 19, Concat 17, Shape 9, Gather 9,
           Unsqueeze 9, MaxPool 6, Transpose 4, Reshape 3, Resize 2
```

A YOLOv7-tiny-family detector. Note the input is *already* NHWC with an explicit leading
`Transpose` to NCHW — so an ideal conversion needs approximately **zero** transposes.

### 4.2 What conversion produces

Intermediate model, straight out of `onnx_to_keras` (channels-first, cannot run on CPU) —
**234 layers**:

```
  58  Conv2D          55  LeakyReLU       53  TFOpLambda      23  ZeroPadding2D
  20  Permute         14  Concatenate      6  MaxPooling2D     4  SlicingOpLambda
   1  InputLayer
```

Final model, after `convert_channels_first_to_last` — **350 layers**:

```
 136  Permute      <-- +116
  58  Conv2D
  55  LeakyReLU
  53  TFOpLambda
  23  ZeroPadding2D
  14  Concatenate
   6  MaxPooling2D
   4  SlicingOpLambda
   1  InputLayer
```

Permutation patterns: `(3,1,2)` ×67, `(2,3,1)` ×66, `(1,3,4,2)` ×3 — i.e. 133 of the 136
are plain 4-D NHWC↔NCHW flips, in near-perfect pairs. Only **6** are back-to-back with
another `Permute` (which is why naive peephole cancellation would barely help — see §6).

Total data physically rewritten by transposes per forward pass at 640×640: **229.3 MB**.

### 4.3 The 57 lambdas are harmless

Every `TFOpLambda`/`SlicingOpLambda` in this model is scalar shape plumbing from the
dynamic-shape detect head (`model.77`) and the two `Resize` nodes. Their output shapes are
`()`, `(1,)`, `(2,)`, `(4,)` and `(5,)`:

```
TFOpLambda  shape             (4,)   model/model.77/Shape_tl_shape              x9
TFOpLambda  gather            ()     model/model.77/Gather_tl_gather            x9
TFOpLambda  expand_dims_v2    (1,)   model/model.77/Unsqueeze_tl_expand_dims    x9
TFOpLambda  concat            (5,)   model/model.77/Concat_tl_concat            x3
TFOpLambda  reshape           (None,None,None,None,None)                        x3
TFOpLambda  cast / multiply / stack / resize_images_v2  (resize scale math)     x~20
SlicingOpLambda  _slice_helper  ()                                              x4
```

Nothing per-pixel. **There is no TFOpLambda explosion in this model.**

### 4.4 Timings (640×640)

```
onnxruntime                        20.1 ms   1.00x
keras tf.function (graph mode)     50.9 ms   2.53x
keras tf.function (dynamic H/W)    56.2 ms   2.80x
keras .predict()                   74.2 ms   3.69x
keras __call__ (eager)             98.2 ms   4.88x
```

**Why the three Keras modes differ.** `model(x)` runs *eagerly* — Python walks all 350
layers, dispatching each op individually; with 350 layers that tax dominates (98.2 vs 50.9
ms, so **~47 ms is pure dispatch overhead**, not compute). `model.predict(x)` compiles the
forward pass into a `tf.function` and so avoids that tax, but adds the data adapter,
batching machinery, callback/distribution plumbing and a numpy→tensor copy per call —
that is the 74.2 vs 50.9 ms gap. `predict` is built for looping over a dataset, not for a
single tensor. **Wrapping the model in `tf.function` directly is the fastest way to run
the h5 as-is.**

### 4.5 Grappler does not fold the transposes

Post-Grappler concrete graph, 867 ops:

```
 236 Const     136 Transpose   117 Placeholder  116 ReadVariableOp
  58 Conv2D     58 BiasAdd      55 LeakyRelu     23 Pad
  17 ConcatV2    9 GatherV2      9 ExpandDims     8 Cast
   6 MaxPool     4 StridedSlice  4 Mul            3 Reshape
   3 Identity    2 Pack          2 ResizeNearestNeighbor   1 NoOp
```

Frozen static-shape graph — still 136 `Transpose`. **All of them survive.** TF's layout
optimiser cannot cancel them because, as written, they are not redundant: the intervening
`LeakyRelu`/`Pad`/`ConcatV2` layers genuinely declare NCHW, so each pair is load-bearing
in the graph even though those ops are themselves layout-agnostic. This is the key point —
the problem cannot be optimised away downstream, it has to be fixed at generation time.

### 4.6 Profiler attribution

Ten graph-mode iterations, leaf ops, CPU-time across threads (see §3.4 caveats). Shares
are of the 111.34 ms of real op time, excluding the `ExecutorState::Process` wrapper:

| op | count/iter | ms/iter | share of op time |
|---|---|---|---|
| `_FusedConv2D` | 58 | 56.42 | **50.7%** |
| `Transpose` | 115 | 33.84 | **30.4%** |
| `LeakyRelu` | 55 | 10.97 | 9.9% |
| `Pad` | 21 | 4.35 | 3.9% |
| `MaxPool` | 6 | 3.52 | 3.2% |
| `ConcatV2` | 17 | 1.95 | 1.8% |
| `ResizeNearestNeighbor` | 2 | 0.23 | 0.2% |
| everything else | — | 0.06 | 0.1% |

Method 1 (standalone re-timing, §3.4) independently gave ≈28%. **Transposes cost roughly
as much as one third of the entire model, and about 60% of what the convolutions cost.**

TF fused `Conv2D`+`BiasAdd` into `_FusedConv2D` but did **not** fuse `LeakyRelu` —
onnxruntime does, which is part of why it stays ahead even after this is fixed.

### 4.7 Cross-check: per-layer-type standalone timings

Independent re-timing of each layer type in isolation, against a 97.9 ms eager baseline:

```
ZeroPadding2D    n= 23      3.4 ms   3.5%
LeakyReLU        n= 55      8.7 ms   8.8%
Conv2D           n= 58     31.9 ms  32.6%
```

Consistent with the profiler ordering. (The script errored out before reaching
`Concatenate`/`MaxPooling2D` on a multi-input layer; those are ≤2% per the profiler and
were not chased.)

### 4.8 Model size

`h5` 25.07 MB vs `onnx` 24.15 MB — weights dominate, no bloat. **File size is not the
issue; graph structure is.**

---

## 5. Survey: 13 local models

Same pipeline, same protocol. `Conv` column counts `Conv*`/`Dense`/`Separable`/`Depthwise`
layers. `Lambdas` counts `TFOpLambda` + `SlicingOpLambda` + `Lambda`.

| model | ONNX nodes | Keras layers | Permute | Conv | Perm/Conv | Lambdas | graph ops | onnx | graph | eager | graph slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **x3d_s** (Conv3D) | 325 | 583 | **236** | 117 | 2.0 | 21 | 1448 | 70.2 ms | 2848.2 ms | 2929.7 ms | **40.6×** |
| **yolo11s** | 320 | 585 | **189** | 88 | 2.1 | 56 | 1472 | 38.9 ms | 149.3 ms | 286.0 ms | **3.84×** |
| **kiwibot** | 222 | 382 | **122** | 47 | 2.6 | 101 | 865 | 25.2 ms | 85.3 ms | 138.8 ms | **3.38×** |
| **yolov7-tiny** | 203 | 374 | **135** | 58 | 2.3 | 24 | 1159 | 26.3 ms | 67.6 ms | 124.6 ms | **2.57×** |
| **v7_5_raw** | 191 | 350 | **136** | 58 | 2.3 | 57 | 867 | 20.1 ms | 50.9 ms | 98.2 ms | **2.53×** |
| dinov2 | 1080 | 583 | 27 | 49 | 0.6 | **407** | 1489 | 18.0 ms | 47.4 ms | 111.3 ms | 2.64× |
| ctformer | 645 | 481 | 15 | 22 | 0.7 | **363** | **11424** | 8.6 ms | 21.0 ms | 163.1 ms | 2.44× |
| rtdetrv2 | 953 | 1766 | 161 | 114 | 1.4 | **1225** | 4106 | 102.1 ms | 137.4 ms | 335.4 ms | 1.35× |
| traffic_light | 1281 | 884 | 130 | 55 | 2.4 | **559** | 2008 | 760.6 ms | 1249.8 ms | 1478.8 ms | 1.64× |
| mnist-12 | 12 | 22 | 8 | 2 | 4.0 | 3 | 50 | 0.02 ms | 0.22 ms | 1.9 ms | 11.2× † |
| infineon (Conv1D) | 96 | 114 | 10 | 10 | 1.0 | 72 | 312 | 0.03 ms | 0.21 ms | 7.8 ms | 6.3× † |
| swin_v2_t ‡ | 6620 | 4369 | 57 | 53 | 1.1 | **3992** | 9691 | 4294.1 ms | 6466.2 ms | 8698.8 ms | 1.51× |
| chip seg ‡ | 44 | 146 | 44 | 14 | 3.1 | 51 | 313 | 693.1 ms | 1529.0 ms | 1708.9 ms | 2.21× |

† sub-millisecond absolute; ratio is dominated by fixed overhead, ignore.
‡ input shape mis-resolved by the harness (§3.5); ratio valid, absolute times are not.

**In every CNN, `graph_transpose` equals or exceeds `n_permute`** — `yolo11s` 189→191,
`dinov2` 27→39, `swin` 57→84, `rtdetrv2` 161→189. Grappler never removes them and
sometimes adds more.

### 5.1 Defect A is universal across CNNs

`yolov7-tiny` is effectively a clone of your model's profile — **135 vs 136 Permutes, 58 vs
58 convs, 2.57× vs 2.53×**. `v7_5_raw` is not a pathological export; this is simply what the
converter does to every convolutional network. The ratio sits at ~2 Permutes per conv
everywhere, which is exactly the signature of a per-layer wrap.

`x3d_s` is the worst case by a wide margin at **40.6×**. 236 Permutes over 5-D NCDHW↔NDHWC
tensors, and unlike the 2-D models it barely benefits from graph mode (2929.7 eager →
2848.2 graph), meaning almost all of its cost is genuine memory traffic rather than
dispatch.

### 5.2 Defect B is universal across transformers

`swin_v2_t` produces **4369 Keras layers with 3992 lambdas** from 6620 ONNX nodes.
`ctformer` turns 645 ONNX nodes into an **11,424-op** graph — a 17× expansion.

The tell is the eager-vs-graph gap, which is pure Python dispatch:

| model | eager | graph | dispatch penalty |
|---|---|---|---|
| infineon | 7.8 ms | 0.21 ms | **37×** |
| test/model.onnx (BERT-like) | 84.0 ms | 8.6 ms | **9.8×** |
| ctformer | 163.1 ms | 21.0 ms | **7.8×** |
| mnist-12 | 1.9 ms | 0.22 ms | 8.6× |
| dinov2 | 111.3 ms | 47.4 ms | 2.3× |
| rtdetrv2 | 335.4 ms | 137.4 ms | 2.4× |

`infineon` is the clean isolation of defect B: only 10 Permutes, but 72 lambdas from 96
ONNX nodes, and a 37× eager penalty with essentially zero real compute.

---

## 6. Root cause

Both the transposes and their non-cancellability come from **two lines** in
`keras-data-format-converter` 0.1.24 — a Tensorleap-owned package
(author `doron.harnoy@tensorleap.ai`), pinned at `pyproject.toml:19`.

**`keras_data_format_converter/layers/layer_utils.py:14`** — only convs and poolings are
ever flipped to channels-last:

```python
onnx_channel_first_cant_run_on_cpu_layers = (Conv, Pooling1D, Pooling2D, Pooling3D)
```

used at `layer_utils.py:30-31`:

```python
if isinstance(current_layer, onnx_channel_first_cant_run_on_cpu_layers):
    config = handle_data_format('channels_last', config)
```

**`keras_data_format_converter/modelconverter.py:172-190`** — and each one is individually
wrapped in a transpose pair:

```python
elif isinstance(converted_layer, onnx_channel_first_cant_run_on_cpu_layers) and \
        current_layer.get_config()['data_format'] == 'channels_first':
    perm_values = calculate_permute_values(node_input_tensors.shape, to_channel_first=False)
    permute_before = tf.keras.layers.Permute(perm_values)(node_input_tensors)
    ...
    tensor = converted_layer(permute_before)
    ...
    perm_values = calculate_permute_values(tensor.shape, to_channel_first=True)
    permute_after = tf.keras.layers.Permute(perm_values)(tensor)
```

### The design is "island conversion"

Each conv/pool becomes an isolated NHWC island inside an otherwise-NCHW graph, and the
tensor is transposed straight back even when the very next layer does not care about
layout. Everything else keeps its original channels-first config — verified on the final
`v7_5_raw` model:

```
conv/pool/padding data_formats: {'channels_last': 64, 'channels_first': 23}
                                                        ^^ all 23 ZeroPadding2D
```

The resulting repeating motif, straight from the converted graph:

```
InputLayer                        (1, None, None, 3)
Permute        dims=(3,1,2)       (1, 3, None, None)
ZeroPadding2D  channels_first     (1, 3, None, None)
Permute        dims=(2,3,1)       (1, None, None, 3)
Conv2D         channels_last      (1, None, None, 32)
Permute        dims=(3,1,2)       (1, 32, None, None)
LeakyReLU                         (1, 32, None, None)
ZeroPadding2D  channels_first     (1, 32, None, None)
Permute        dims=(2,3,1)       (1, None, None, 32)
Conv2D         channels_last      (1, None, None, 64)
Permute        dims=(3,1,2)       (1, 64, None, None)
LeakyReLU                         (1, 64, None, None)
...
```

repeated 58 times. This yields exactly the 2-Permutes-per-conv ratio measured in every
model in §5.

**Why a naive peephole pass will not fix it:** the inverse Permutes are separated by
`LeakyReLU` and `ZeroPadding2D`, so only **6 of 136** pairs in `v7_5_raw` are actually
adjacent. Cancelling only adjacent pairs would recover ~4% of the transposes. The
intervening layers have to be made layout-transparent first.

---

## 7. Approved fix plan

Decisions taken:

- **Where:** in `keras-data-format-converter` (the source), not as a post-pass in
  `onnx2kerastl`. It is the correct home, benefits every consumer, avoids duplicating
  layout logic, and avoids a second full graph rebuild on every conversion. Requires
  releasing that repo and bumping the pin at `pyproject.toml:19` from `0.1.24`.
- **Scope:** **2-D CNN path only** for the first change. Covers `v7_5_raw`, `yolov7-tiny`,
  `yolo11s`, `kiwibot`, `chip`, `traffic_light`. `Conv1D`/`Conv3D` (and therefore
  `x3d_s`, the 40.6× worst case) stay on today's path for now.

### 7.1 Approach: layout propagation instead of island wrapping

Keep the tensor in channels-last after a conv and **propagate the layout forward**,
inserting a transpose only where a layer genuinely requires channels-first. Classify each
layer into three buckets:

1. **Layout-agnostic** — pass through untouched, carry the layout tag forward.
   `Activation`, `ReLU`, `LeakyReLU`, `PReLU`, `ELU`, `Add`, `Multiply`, `Subtract`,
   `Maximum`, `Minimum`, `Dropout`, and elementwise `TFOpLambda`.

2. **Layout-adaptable** — rebuild with one config field remapped, then carry the tag.
   `ZeroPadding2D` / `Cropping2D` / `UpSampling2D` (`data_format`), `BatchNormalization` /
   `Concatenate` / `Softmax` (`axis`), `GlobalAveragePooling2D` / `GlobalMaxPooling2D`
   (`data_format`).
   *Note: for `ZeroPadding2D` the padding tuple itself is unchanged — it is
   `((top,bottom),(left,right))` in both formats — only `data_format` moves.*

3. **Layout-fixed or unknown** — insert a transpose back to channels-first, exactly as
   today. `Reshape`, `Dense`, `TFOpLambda` with baked-in axes, the ONNX detect head, and
   **all model outputs**.

For a CNN backbone this collapses the transposes to a handful at the head boundary. For
`v7_5_raw` specifically, whose ONNX input is already NHWC, the ideal is near zero.

### 7.2 Implementation sketch

In `modelconverter.py`:

- Track a layout tag per converted tensor, e.g. `self._tensor_layout: Dict[int, str]`
  alongside the existing `self._tensor_cache`.
- Add `LAYOUT_AGNOSTIC_2D` and `LAYOUT_ADAPTABLE_2D` registries in `layer_utils.py`
  beside the existing `onnx_channel_first_cant_run_on_cpu_layers`.
- In `_convert_tensor`, before invoking a layer: determine its required layout, and emit a
  `Permute` **only** when the incoming tag differs from what is required.
- Multi-input layers (`Concatenate`, `Add`): reconcile input tags. If they disagree,
  transpose the minority to match — or, for the conservative first cut, force all inputs
  to channels-first and fall back to today's behaviour.
- In `convert_model`, force every output tensor back to the original layout before
  building the `keras.Model`, so the public contract is unchanged.
- Interaction with `should_transform_inputs_and_outputs` must be preserved; both settings
  need coverage.

### 7.3 Expected result (estimate, not yet measured)

From the §4.6 profile: removing substantially all transposes should take ~30% off
graph-mode wall time, **51 → ~36 ms**, and drop the layer count **350 → ~215**, which cuts
eager dispatch roughly proportionally, **98 → ~62 ms**. That moves `v7_5_raw` from 2.53× to
roughly **1.8×** onnxruntime.

It will **not** reach parity. onnxruntime additionally fuses `Conv`+`LeakyRelu` (TF fuses
only `Conv`+`BiasAdd`, per §4.6) and uses better convolution kernels. Closing that
remaining gap is out of scope here.

### 7.4 Correctness gate

This changes **every** converted model, so it needs:

- Numeric equivalence vs onnxruntime across the `test/` corpus — the 13 models in §5 make
  a ready-made regression set, with `2.3e-05` as the demonstrated achievable tolerance.
- Layer-count and `graph_transpose` assertions to prove the transposes actually went away
  and to catch regressions.
- Both `should_transform_inputs_and_outputs=True` and `False`.
- Explicit check that 1-D and 3-D models (`infineon`, `x3d_s`) are **byte-identical** to
  today's output, since they are out of scope for this change.

---

## 8. Follow-ups not covered by this plan

1. **Defect B — eager dispatch.** Largely not an `onnx2keras` problem: Tensorleap runs
   these h5 models eagerly. Wrapping the loaded model in `tf.function` at inference time
   in the engine / code-loader would recover **2×–37×** on the transformer models for
   free, with no conversion changes at all. Worth raising against those repos separately.
   This is probably the single highest value-per-effort item in this document.

2. **Defect B — lambda count itself.** `swin` at 3992 lambda layers and `ctformer`'s 17×
   node→op expansion suggest the ONNX→Keras op mapping emits far more ops than necessary
   for transformer patterns. Separate investigation.

3. **`Conv1D`/`Conv3D` layout propagation** — deferred out of §7 scope, but `x3d_s` at
   **40.6×** is the worst offender measured and should be picked up next.

4. **Unrelated bug on branch `fix-bug-reshap-layers`.** `v7_5_raw` fails to convert there
   entirely: `onnx2kerastl/reshape_layers.py:260` calls `K.is_keras_tensor()` on a raw
   `np.ndarray`, which raises `ValueError: Unexpectedly found an instance of type
   <class 'numpy.ndarray'>. Expected a symbolic tensor instance.` The guard needs the
   `tf.is_tensor(inp)` check reordered ahead of the `K.is_keras_tensor(inp)` call.
   `master` is unaffected. All measurements in this document were taken on `master`
   @ `b54841e`.
