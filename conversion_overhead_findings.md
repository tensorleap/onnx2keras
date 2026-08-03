# ONNX → Keras conversion overhead: findings and fix plan

Investigation of whether `onnx2kerastl` produces "overblown" Keras models that make
inference slow. Triggered by `overwatch/model/v7_5_raw.onnx`, then generalised to a
12-model sample of the ~22 ONNX models held in this repo (see §5 for what was left out
and why).

**Status:** findings validated, fix plan approved, implementation not started. GPU
reproduction for `v7_5_raw` is now done — see §9 — and it confirms the §1 prediction:
defect A shrinks on GPU, defect B grows. The other 12 surveyed models are not yet
re-run on GPU.

A critical review of this document is in
[`conversion_overhead_findings_review.md`](conversion_overhead_findings_review.md). Its
findings on the Defect B causal story (§5.2 here), the transpose share (§4.6), survey
scope (§5), and the shape-lambda gap in the fix plan (§7.1) have been accepted and folded
in.

---

## 1. TL;DR

There are **two independent defects**, both general (not specific to any one model):

| # | Defect | Who it hits | Cost |
|---|---|---|---|
| **A** | **Transpose sandwich** — every `Conv`/`Pooling` layer is wrapped in its own `Permute`-in/`Permute`-out pair, so the tensor ping-pongs NHWC↔NCHW around every convolution | every CNN, without exception (~2.0–2.6 Permutes per conv) | est. **30–50% of graph-mode wall time** (§4.6 — an interval, not a measurement); 2.5× slower than onnxruntime on `v7_5_raw`. `x3d_s` is 40.6× but its attribution to this defect is **unvalidated** (§5.1) |
| **B** | **Eager dispatch, inflated by lambda explosion** — running a converted model eagerly costs **~0.1–0.3 ms per Keras layer** in Python dispatch, and transformer graphs expand into thousands of `TFOpLambda`/`Lambda` layers | anything with a high layer count; worst is `swin` at 4369 layers / 3992 lambdas | **~2.2 s per call** on `swin`, ~142 ms on `ctformer`. Note the eager/graph *ratio* does **not** track lambda count (§5.2) |

For the originating question — **`v7_5_raw` is defect A, not B.** Its 57 lambda layers are
all scalar shape plumbing (`Shape`/`Gather`/`Unsqueeze` on `()`- and `(4,)`-shaped tensors)
and cost essentially nothing. The 136 `Permute` layers are the problem.

Conversion is **numerically correct** in all cases checked — max abs diff vs onnxruntime
on `v7_5_raw` is `2.3e-05` across all three outputs.

> ### ⚠ Sections 1–8 below are CPU. Tensorleap's usual target is GPU.
>
> These conclusions were established on CPU. §9 now confirms the prediction below on one
> model (`v7_5_raw`, one box) — the other 12 surveyed models are still unconfirmed on GPU.
>
> - cuDNN natively prefers **NCHW** — the opposite of CPU — so TF may keep convolutions
>   channels-first and emit fewer transposes, or none of the per-conv pattern at all.
> - Transposes are bandwidth-bound and GPU bandwidth is far higher, so defect A's share
>   should **fall**.
> - Kernels get much faster while Python dispatch cost does not, so defect B's
>   eager-vs-graph penalty should get **relatively worse**.
>
> Net expectation: **defect A shrinks on GPU, defect B grows.** §9 confirms this for
> `v7_5_raw`: graph-mode slowdown 2.90x (CPU) → 1.19x (GPU), eager slowdown 7.45x (CPU) →
> 21.5x (GPU). The fix priority in §7 **does invert** — §8.1 (wrapping models in
> `tf.function` in the engine) is the first thing to do. Re-run `benchmarks/` on GPU for
> the remaining 12 models before committing further effort to the §7 converter work.

Reproduce anything in this document with the tooling in
[`benchmarks/`](benchmarks/) — see [`benchmarks/README.md`](benchmarks/README.md).

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

Modelled on `onnx2kerastl/convert_model.py:8-26`:

```python
onnx_model = onnx.load(path)
initializers = {n.name for n in onnx_model.graph.initializer}
input_names = [i.name for i in onnx_model.graph.input if i.name not in initializers]
keras_model = onnx_to_keras(onnx_model, input_names=input_names,
                            name_policy='attach_weights_name',
                            allow_partial_compilation=False).converted_model
final_model = convert_channels_first_to_last(keras_model,
                                             should_transform_inputs_and_outputs=False)
```

The second step matters: `onnx_to_keras` alone emits a **channels-first** model, which
cannot run on CPU at all (`Conv2D op currently only supports the NHWC tensor format on the
CPU`). So every measurement below is of the post-`convert_channels_first_to_last` model.

> **Two deliberate departures from `convert_model.py`.** That script filters nothing from
> `graph.input` (initializers included) and passes `verbose=True`; the harness filters
> initializers and runs quiet. Neither affects the resulting graph.
>
> **One that needed checking: `should_transform_inputs_and_outputs`.** The library default
> is `False` (`converterapi.py:14`) and that is what is measured here, but
> `convert_model.py:8` defaults its own `transform_io` to `True`, so this is not that
> script's default path. Measured both on `v7_5_raw`:
>
> ```
> transform_io=False: 350 layers, 136 Permute, 867 graph ops, maxdiff 1.9e-05
> transform_io=True:  354 layers, 140 Permute, 875 graph ops, input (1,None,3,None)
> ```
>
> The flag adds **4 boundary transposes** and leaves the 136 internal ones untouched, so
> it does not affect any conclusion here. It also transposes the model's I/O: since this
> model's ONNX input is already NHWC, `True` yields a model that rejects the natural input
> tensor with a shape error and returns transposed outputs. `False` is the correct setting
> for it. Which setting Tensorleap actually invokes in production is still worth
> confirming, and §7.2 commits to preserving both.

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

### 3.4 Per-op cost attribution (the transpose share)

Three methods. **They do not agree, and an earlier revision of this document wrongly
claimed they did** — see §4.6 for the reconciled range.

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

Result for `v7_5_raw`: **26.1 ms** of graph-mode wall time across 136 Permutes (20
iterations each).

> **Denominator warning.** An earlier revision divided this by the 94.7 ms full-model
> *eager* time to get ≈28%, which is wrong: the numerator is graph-mode wall time and the
> eager denominator contains ~47 ms of Python dispatch that the numerator does not (§4.4).
> Against the quantity a fix would actually reduce — graph-mode wall time, 50.9 ms —
> Method 1 gives **26.1 / 50.9 ≈ 51%**.
>
> Method 1 also over-attributes: standalone re-timing pays full call-entry and cold-cache
> cost per op and loses in-graph locality. Held to a consistent eager baseline, §4.7's
> layer types (44.0 ms) plus these Permutes (26.1 ms) plus dispatch (47.3 ms) total
> 117.4 ms against a measured 98.2 ms — **120% of runtime**, before MaxPool, ConcatV2,
> BiasAdd and 57 lambdas. So ~51% is an upper bound, not a point estimate.

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
>
> **A thread-time share is not a wall-time share.** Converting one to the other assumes
> every op parallelises equally well, which is false here — see Method 3.

**Method 3 — single-threaded profiler.** The same profiler run with
`tf.config.threading.set_intra_op_parallelism_threads(1)` and
`set_inter_op_parallelism_threads(1)`. At one thread, thread-time is wall-time, so the
share is directly interpretable — confirmed by the leaf-op sum landing at 232.2 ms
against 238.0 ms wall, a ratio of **0.98**:

```
1 thread: onnx 22.4 ms | keras graph 238.0 ms | 10.62x
  _FusedConv2D    58    163.27 ms   70.3%
  Transpose      115     30.78 ms   13.3%
  LeakyRelu       55     30.00 ms   12.9%
  Pad             21      3.49 ms    1.5%
  MaxPool          6      2.50 ms    1.1%
  ConcatV2        17      1.96 ms    0.8%
```

This does **not** simply resolve the multi-thread question — it answers a different one,
giving 13.3% under single-thread conditions that nobody deploys. Its real value is the
comparison against Method 2: `Transpose` costs ~31–34 ms at both 1 and 16 threads
(**parallelism ≈ 1.1×, effectively serial**), while `_FusedConv2D` drops from 163.27 ms
of single-thread work to 56.42 ms of summed thread-time. Convolutions parallelise;
transposes do not.

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

> **The harness has since been fixed** (`benchmarks/common.py:resolve_dim`): axis 0 now
> resolves to 1 unconditionally, regardless of its symbolic name. The §5 numbers were
> produced *before* that fix and are reported as measured; re-running the survey will
> change the `swin` and `chip` rows.

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

A repeat of the same run on a more loaded machine (onnxruntime 20.1 → 32.3 ms) gave
`_FusedConv2D` 56.6% and `Transpose` **22.4%**.

#### Reconciling the methods

These numbers measure different things and an earlier revision of this document wrongly
presented them as mutual corroboration. What each one actually says:

| method | quantity | `Transpose` |
|---|---|---|
| 2 — profiler, 16 threads | share of **thread-time** | 22–30% |
| 3 — profiler, 1 thread | share of **wall time, single-threaded** | 13.3% |
| 1 — standalone re-timing | graph wall / graph wall, **over-attributes** | ≤51% |

The multi-threaded *wall* share — the quantity that matters — is bounded rather than
measured. Total op thread-time is 111.34 ms against ~51 ms wall, so average parallelism
is 2.18×. If `Transpose` parallelised at that average its wall share would be ~30%; if it
were fully serial, 33.84 ms of thread-time is 33.84 ms of wall time, or ~66%.

Method 3 settles which end applies: **transposes are effectively serial** (~31–34 ms of
work at both 1 and 16 threads) while convolutions parallelise well. That pushes the true
value toward the upper part of the interval, and Method 1's upper bound of ~51% is
consistent with it.

> **Best current estimate: transposes are ~30–50% of multi-threaded graph wall time.**
> The 30.4% figure carried by earlier revisions of this document is the **floor**, not the
> estimate. The error direction favours the fix.
>
> This remains an interval, not a measurement. The only decisive test is a prototype with
> the transposes actually removed, timed against the current model — no profiler
> configuration substitutes for it. Treat §7.3's projection accordingly.

TF fused `Conv2D`+`BiasAdd` into `_FusedConv2D` but did **not** fuse `LeakyRelu` —
onnxruntime does, which is part of why it stays ahead even after this is fixed.

#### Unexplained op-count gap

The profiler reports 115 `Transpose` and 21 `Pad` per iteration, but §4.2 counts 136
`Permute` and 23 `ZeroPadding2D`, and §4.5 confirms all 136 survive into the frozen
graph. **21 transposes and 2 pads are unaccounted for.** Either they execute on
constant/weight paths and are folded at runtime — in which case "136 Permutes" overstates
the runtime problem by ~15% — or the profiler is dropping events, in which case the
shares above are understated. Not yet resolved; it should be before §7.3's projection is
relied on.

### 4.7 Sanity check only: per-layer-type standalone timings

Independent re-timing of each layer type in isolation, against a 97.9 ms eager baseline:

```
ZeroPadding2D    n= 23      3.4 ms   3.5%
LeakyReLU        n= 55      8.7 ms   8.8%
Conv2D           n= 58     31.9 ms  32.6%
```

This reproduces the profiler's *ordering* and nothing more, which is close to zero
information — two methods both ranking convolution first was never in doubt. It is not
quantitative corroboration, and these shares suffer the same over-attribution as Method 1
(§3.4). The original run also errored out before reaching `Concatenate`/`MaxPooling2D` on
a multi-input layer; that bug is fixed in `benchmarks/layer_costs.py`, and those types are
≤2% per the profiler.

### 4.8 Model size

`h5` 25.07 MB vs `onnx` 24.15 MB — weights dominate, no bloat. **File size is not the
issue; graph structure is.**

---

## 5. Survey: 13 surveyed models (12 local + `v7_5_raw`)

Same pipeline, same protocol, `--iters 10 --warmup 3`. `Conv` column counts
`Conv*`/`Dense`/`Separable`/`Depthwise` layers. `Lambdas` counts `TFOpLambda` +
`SlicingOpLambda` + `Lambda`.

> **This is a sample, not the corpus.** The repo holds ~22 unique ONNX models (29 files,
> several duplicated between `./` and `./test/`). Not surveyed: `maskrcnn`, `clip`,
> `lung_anatomy_merged`, `lung_anatomy_merged_ir9`, `mmdet_convnext`, `rfdetr-base`,
> `interfuser_planKD`, `TrackerInferenceFcudarc3aug`, `raft`, `nms_v2`, `split_model`.
>
> They were excluded for **size and runtime** — most are 100–250 MB — **not because they
> fail to convert**; that was never tested. The omission is not random and it matters:
> `maskrcnn`, `nms_v2`, `raft` and `split_model` are the control-flow, NMS and
> dynamic-shape graphs where the §7.1 layout rewrite is most likely to break. §7.4 leans
> on this survey as ready-made regression coverage, so those models need converting at
> least once before that claim holds.

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
| test/model.onnx (BERT-like) | 1183 | 1190 | 48 | 74 | 0.6 | **801** | 2068 | 2.2 ms | 8.6 ms | 84.0 ms | 3.87× † |
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

`yolov7-tiny` is effectively a clone of `v7_5_raw`'s structure — **135 vs 136 Permutes
over 58 vs 58 convs**. Layer counts are deterministic, so that comparison is exact.
(Their slowdowns of 2.57× and 2.53× are *not* evidence of anything: the 1.6% difference
sits far inside the ±10–20% variance disclosed in §2. The structural counts carry the
argument alone.)

`v7_5_raw` is not a pathological export; this is simply what the converter does to every
convolutional network. The ratio sits at ~2 Permutes per conv everywhere, which is
exactly the signature of a per-layer wrap.

`x3d_s` is the worst case by a wide margin at **40.6×**, with 236 Permutes over 5-D
NCDHW↔NDHWC tensors.

> **Unvalidated: `x3d_s` has never been profiled.** Attributing its 40.6× to Defect A
> rests only on `eager ≈ graph` (2929.7 → 2848.2 ms), which shows the cost is not Python
> dispatch — it does not show the cost is transposes. TF's `Conv3D` kernels are
> independently far slower than onnxruntime's, and that alternative has not been ruled
> out. If most of the 40.6× is kernel quality, layout propagation will not fix it and the
> §8.3 prioritisation is wrong. One `benchmarks/profile_ops.py` run settles it.

### 5.2 Defect B: dispatch cost scales with layer count, and lambdas inflate layer count

`swin_v2_t` produces **4369 Keras layers with 3992 lambdas** from 6620 ONNX nodes.
`ctformer` turns 645 ONNX nodes into an **11,424-op** graph — a 17× expansion.

> **Correction.** An earlier revision framed this section around the eager-vs-graph
> *ratio* and claimed lambda count predicts it. **It does not.** That framing also
> presented a six-row table that omitted the three rows contradicting it, and led with
> `infineon`, a row this document had already told the reader to ignore. Both are
> corrected below.

Eager/graph ratio for **every** surveyed row, ordered by lambda count:

| lambdas | model | eager | graph | ratio |
|---|---|---|---|---|
| 3992 | swin_v2_t | 8698.8 ms | 6466.2 ms | **1.35×** |
| 1225 | rtdetrv2 | 335.4 ms | 137.4 ms | 2.44× |
| 801 | test/model.onnx | 84.0 ms | 8.6 ms | 9.8× |
| 559 | traffic_light | 1478.8 ms | 1249.8 ms | **1.18×** |
| 407 | dinov2 | 111.3 ms | 47.4 ms | 2.35× |
| 363 | ctformer | 163.1 ms | 21.0 ms | 7.8× |
| 101 | kiwibot | 138.8 ms | 85.3 ms | 1.63× |
| 72 | infineon | 7.8 ms | 0.21 ms | 37× |
| 3 | mnist-12 | 1.9 ms | 0.22 ms | 8.6× |

**The two models with the most lambdas have the smallest ratios.** The asserted
correlation is absent, arguably negative.

What the data does support is a flat per-layer cost. `(eager − graph) / layer_count`:

| model | µs/layer | | model | µs/layer |
|---|---|---|---|---|
| infineon | 67 | | kiwibot | 140 |
| mnist-12 | 76 | | x3d_s | 140 |
| dinov2 | 110 | | yolov7-tiny | 152 |
| rtdetrv2 | 112 | | yolo11s | 234 |
| test/model.onnx | 63 | | traffic_light | 259 |
| v7_5_raw | 135 | | ctformer | 295 |
| | | | swin_v2_t | 511 |

(`chip` is an outlier at ~1230 µs/layer, consistent with its mis-resolved input per §3.5.)

**Eager execution costs roughly 0.1–0.3 ms per Keras layer in Python dispatch.** The
*ratio* varies 1.18×–37× only because it measures how little real compute a model has
relative to its layer count — `infineon`'s 37× is a 7.6 ms absolute saving on a
sub-millisecond model.

Lambda explosion still matters, but through layer count rather than through the ratio: at
4369 layers, `swin_v2_t` pays **~2.2 seconds per call** in dispatch. That is the argument
for §8.1, and it has to be made in milliseconds.

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

> **Gated on GPU confirmation.** This plan was approved against CPU evidence. Per the
> banner in §1, defect A may be much smaller on GPU, which is the usual deployment
> target. Re-run `benchmarks/survey.py` and `benchmarks/profile_ops.py` on GPU first; if
> the Permute share collapses there, do §8.1 instead of this.

Decisions taken:

- **Where:** in `keras-data-format-converter` (the source), not as a post-pass in
  `onnx2kerastl`. It is the correct home, benefits every consumer, avoids duplicating
  layout logic, and avoids a second full graph rebuild on every conversion. Requires
  releasing that repo and bumping the pin at `pyproject.toml:19` from `0.1.24`.
  Repo: `git@github.com:tensorleap/keras-data-format-converter.git`
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

3. **Shape-consuming** — any op that reads a tensor's *dimensions* rather than its
   values: `Shape`, `Gather`/`Unsqueeze`/`Concat` on the shape path, and anything feeding
   a dynamic `Reshape` target. **Must see the original layout**; transpose back before it.

4. **Layout-fixed or unknown** — insert a transpose back to channels-first, exactly as
   today. `Reshape`, `Dense`, `TFOpLambda` with baked-in axes, the ONNX detect head, and
   **all model outputs**.

> **Bucket 3 is the trap, and an earlier revision of this plan omitted it.** A `Shape` op
> is neither elementwise (bucket 1) nor axis-baked (bucket 4), so a three-bucket taxonomy
> silently routes it into "pass through and carry the tag". It would then read **NHWC**
> dimensions where the consumer expects NCHW — producing a wrong reshape target rather
> than a crash. That is **silent numerical corruption**, the worst possible failure mode
> for this change.
>
> `v7_5_raw` has exactly this pattern: 9 `Shape` lambdas feeding `model.77`, whose outputs
> build the three detect-head `Reshape` targets (§4.3). Any implementation must trace
> shape provenance, not just op type. The §7.4 numeric gate would catch it — but only if
> the affected models are in the gate, which is another reason §5's excluded models matter.

For a CNN backbone this collapses the transposes to a handful at the head boundary. For
`v7_5_raw` specifically, whose ONNX input is already NHWC, the ideal is near zero.

### 7.2 Implementation sketch

In `modelconverter.py`:

- Track a layout tag per converted tensor, e.g. `self._tensor_layout: Dict[int, str]`
  alongside the existing `self._tensor_cache`.
- Add `LAYOUT_AGNOSTIC_2D` and `LAYOUT_ADAPTABLE_2D` registries in `layer_utils.py`
  beside the existing `onnx_channel_first_cant_run_on_cpu_layers`.
- Add shape-provenance tracking for bucket 3: mark any tensor derived from a `Shape` op,
  and force its producer's input back to the original layout. Default any unrecognised
  `TFOpLambda` into bucket 4 rather than bucket 1, so the failure mode is a redundant
  transpose rather than a wrong result.
- In `_convert_tensor`, before invoking a layer: determine its required layout, and emit a
  `Permute` **only** when the incoming tag differs from what is required.
- Multi-input layers (`Concatenate`, `Add`): reconcile input tags. If they disagree,
  transpose the minority to match — or, for the conservative first cut, force all inputs
  to channels-first and fall back to today's behaviour.
- In `convert_model`, force every output tensor back to the original layout before
  building the `keras.Model`, so the public contract is unchanged.
- Interaction with `should_transform_inputs_and_outputs` must be preserved; both settings
  need coverage.

### 7.3 Expected result

An earlier revision projected "~30% off graph-mode wall time, 51 → ~36 ms". **That
projection has been withdrawn**: it converted a thread-time share directly into a
wall-time saving, which §4.6 shows is not valid.

What can be said:

- **Layer count 350 → ~215** for `v7_5_raw`. This one is structural and reliable.
- **Eager time should fall roughly proportionally** at ~0.1–0.3 ms/layer (§5.2): ~135
  layers removed ≈ **18 ms** off the 98.2 ms eager figure.
- **Graph-mode saving is bounded, not predicted** — somewhere inside the ~30–50% interval
  of §4.6, and transposes being effectively serial (§3.4 Method 3) argues for the upper
  part of it. Anything more precise requires the prototype.

It will **not** reach parity regardless. onnxruntime additionally fuses `Conv`+`LeakyRelu`
(TF fuses only `Conv`+`BiasAdd`, per §4.6) and uses better convolution kernels. Closing
that remaining gap is out of scope here.

### 7.4 Correctness gate

This changes **every** converted model, so it needs:

- Numeric equivalence vs onnxruntime, with `2.3e-05` as the demonstrated achievable
  tolerance. The 13 models in §5 are a starting point, **not** ready-made coverage: the
  ~10 unsurveyed models (§5) include the control-flow and dynamic-shape graphs most at
  risk, and they must be converted at least once before this gate means anything.
- **A model with a dynamic-shape head in the gate** — `v7_5_raw` itself, or any model with
  `Shape`-derived `Reshape` targets — since bucket 3 (§7.1) fails silently and numerics
  are the only thing that catches it.
- Layer-count and `graph_transpose` assertions to prove the transposes actually went away
  and to catch regressions.
- Both `should_transform_inputs_and_outputs=True` and `False`, now that §3.1 has baselined
  the difference (4 boundary transposes).
- Explicit check that 1-D and 3-D models (`infineon`, `x3d_s`) are **byte-identical** to
  today's output, since they are out of scope for this change.
- **Downstream layer-graph stability.** Removing ~116 `Permute` layers changes the graph
  Tensorleap displays and maps analyses onto, and changes layer names under
  `name_policy='attach_weights_name'`. Saved analyses, visualisations and anything keyed
  on layer identity may break. This is a product-visible side effect, not just a
  numerical one, and it needs an owner outside this repo before the change ships.
  (**Unvalidated** — no downstream consumer was inspected.)

---

## 8. Follow-ups not covered by this plan

1. **Defect B — eager dispatch.** Largely not an `onnx2keras` problem: Tensorleap runs
   these h5 models eagerly. Wrapping the loaded model in `tf.function` at inference time
   in the engine / code-loader costs no conversion changes at all and saves, in absolute
   terms per call:

   | model | saving | | model | saving |
   |---|---|---|---|---|
   | swin_v2_t | **2233 ms** | | dinov2 | 64 ms |
   | traffic_light | **229 ms** | | test/model.onnx | 75 ms |
   | rtdetrv2 | **198 ms** | | kiwibot | 54 ms |
   | ctformer | 142 ms | | v7_5_raw | 47 ms |

   Stated in milliseconds rather than as the ratio an earlier revision used — see the
   correction in §5.2. This is still the highest value-per-effort item in this document,
   and **on GPU it likely outranks §7 outright**: dispatch is fixed cost that does not
   shrink as kernels get faster, so its relative share grows. Confirm with a GPU run of
   `benchmarks/survey.py`, comparing `t_keras_eager` against `t_keras_graph`.

2. **Defect B — lambda count itself.** `swin` at 3992 lambda layers and `ctformer`'s 17×
   node→op expansion suggest the ONNX→Keras op mapping emits far more ops than necessary
   for transformer patterns. At ~0.1–0.3 ms/layer this is where the dispatch cost is
   manufactured, so reducing it compounds with item 1. Separate investigation.

3. **`Conv1D`/`Conv3D` layout propagation** — deferred out of §7 scope. `x3d_s` at
   **40.6×** is the worst ratio measured, but per §5.1 that has **not** been attributed to
   Defect A. Profile it before prioritising: if TF's `Conv3D` kernels rather than layout
   are the cause, this item is worthless and should be dropped.

4. **Resolve the 115-vs-136 transpose gap** (§4.6) — decides whether the runtime problem
   is ~15% smaller than the layer count suggests, or the profiler shares are understated.

5. **Verify the h5 round-trip.** Every measurement here is of the in-memory converted
   model. Production saves an h5 and §8.1 notes Tensorleap loads and runs that file; the
   two need not be the same graph, and `layer_utils.py:24` shows revived layers take a
   distinct path through the converter. One save/load/re-measure closes it.
   (**Unvalidated** — no evidence the h5 differs in practice.)

4. **Unrelated bug on branch `fix-bug-reshap-layers`.** `v7_5_raw` fails to convert there
   entirely: `onnx2kerastl/reshape_layers.py:260` calls `K.is_keras_tensor()` on a raw
   `np.ndarray`, which raises `ValueError: Unexpectedly found an instance of type
   <class 'numpy.ndarray'>. Expected a symbolic tensor instance.` The guard needs the
   `tf.is_tensor(inp)` check reordered ahead of the `K.is_keras_tensor(inp)` call.
   `master` is unaffected. All measurements in this document were taken on `master`
   @ `b54841e`.

---

## 9. GPU validation — `v7_5_raw` only

Runs the §1 banner's prediction against real GPU hardware, on the one model available
locally on this box at the time. **This is one model, one GPU, one run each** — it
confirms the *direction* of the CPU→GPU prediction, not a full re-survey. The other 12
models in §5 are still unconfirmed on GPU; re-running them is the natural next step and
should use the same `LD_LIBRARY_PATH` fix below.

### 9.1 Environment

```
platform        Linux (Ubuntu), x86_64
GPU             NVIDIA A10G, driver 580.126.09, compute capability 8.6
tensorflow      2.12.0   (same version as §2)
onnxruntime-gpu 1.17.1
cuDNN           8.6.0 (via pip nvidia-cudnn-cu11)
CUDA (pip)      11.8 runtime/cublas (via nvidia-cuda-runtime-cu11 / nvidia-cublas-cu11)
CUDA (system)   12.6 / 12.8 / 12.9 / 13.0 toolkits under /usr/local
```

**Gotcha that will bite the next person running this.** `nvidia-smi` on this box shows a
working A10G, but `tf.config.list_physical_devices('GPU')` returned `[]` and onnxruntime
silently fell back to `CPUExecutionProvider`, out of the box. Not a missing-driver
problem — TF 2.12 / onnxruntime-gpu 1.17 are built against **CUDA 11.x**, and their
matching `libcublasLt.so.11` / `libcudart.so.11.0` ship as pip packages
(`nvidia-cublas-cu11`, `nvidia-cuda-runtime-cu11`, already in this project's poetry lock).
Those `.so` files exist, under
`<venv>/lib/python3.10/site-packages/nvidia/*/lib/`, but the shell's
`LD_LIBRARY_PATH` pointed only at the system-wide **CUDA 12.9** install, which has no
`.so.11` files, and the dynamic linker never looked in the venv. Fix:

```bash
VENV=$(poetry env info --path)
SITE=$VENV/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$(find $SITE/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH}"
```

After this, `tf.config.list_physical_devices('GPU')` reports the device, and
`ort.InferenceSession(..., providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
.get_providers()` confirms `CUDAExecutionProvider` is actually selected (checked
explicitly — a provider being *available* is not the same as a session *using* it).

### 9.2 Structural counts reproduce exactly

Independent of CPU vs GPU — conversion happens once, before either backend runs the
graph. On this box: **350 Keras layers, 136 Permute, 58 Conv2D, 57 lambdas** — identical
to §4.2's 350/136/58/57. Same numeric-correctness result too (finite, matching output
shapes on all three heads).

### 9.3 Timings (640×640, `benchmarks.survey` / `benchmarks.profile_ops`)

```
                    CPU (this box)   GPU (this box)
onnxruntime              28.8 ms          4.7 ms
keras graph              83.5 ms          5.6 ms   (0.99x in a separate profiler run)
keras eager             214.7 ms        100.7 ms
slowdown, graph          2.90x            1.19x
slowdown, eager          7.45x           21.5x
```

(The CPU column is this box, not §4.4's macOS numbers — included here only to show the
CPU→GPU *shift* on one consistent machine. It is in the same ballpark as §4.4: 2.90x vs
2.53x graph, 7.45x vs 4.88x eager, within the ±10–20% cross-machine variance §2
discloses.)

**This confirms both halves of the §1 prediction:**

- **Defect A collapses.** Graph-mode slowdown drops from 2.90x to essentially parity
  (1.19x, one run as low as 0.99x). onnxruntime's convolutions speed up on GPU by roughly
  the same factor as Keras's do, so the transpose tax that dominated the CPU gap becomes
  a rounding error against ~5 ms of wall time.
- **Defect B gets much worse.** Eager slowdown nearly triples, 7.45x → 21.5x. Python
  dispatch cost per layer does not fall with faster kernels, so at GPU speeds it goes
  from "a meaningful tax" to "the entire story" — the model has 350 layers regardless of
  backend, and every one of them still pays a Python round-trip in eager mode.

`profile_ops.py` on GPU (leaf events on `/device:GPU:0`, thread-time not wall-time, same
caveat as §3.4 Method 2): the transpose-family kernel (`...Dimension<3>, unsigned int...`
— cuDNN/XLA event names are truncated by the profiler's metadata table, but the op count
is 115, matching §4.6's `Transpose` count exactly) is **~23% of device time**, in the same
range as §4.6's multi-thread CPU estimate (22–30%). The *kernel-level* attribution barely
moved; what moved is that total wall time shrank from ~83 ms to ~5 ms, so the same
percentage share is a much smaller absolute cost, and onnxruntime's GPU kernels got
proportionally faster too.

### 9.4 New finding: GPU numeric tolerance is not the CPU tolerance

§7.4 proposes `2.3e-05` (§1, CPU) as the correctness-gate tolerance. On GPU it does not
hold:

```
TF32 enabled (TF default):   max abs diff vs onnxruntime ≈ 0.025–0.029 across p3/p4/p5
TF32 disabled (fp32 exact):  max abs diff vs onnxruntime ≈ 0.008–0.010
```

Disabling TF32 (`tf.config.experimental.enable_tensor_float_32_execution(False)`) cuts
the diff by ~3× but does not close it to CPU levels — the remainder is ordinary
fp32 accumulation-order divergence compounding across 58 convolutions on different
kernels (cuDNN vs onnxruntime's own GPU kernels), not a bug. **Any §7.4 gate that runs on
GPU needs its own, looser tolerance** — `2.3e-05` was only ever demonstrated on CPU.

### 9.5 What this does and doesn't settle

- It validates the §1 banner's *direction* on the one model this box had available.
  Re-running §5's other 12 models on GPU (with the `LD_LIBRARY_PATH` fix above) is the
  natural next step and would turn this from "one model confirms the prediction" into
  "the survey confirms it."
- It makes item 8.1 (`tf.function` wrapping in the engine) look more urgent, not less —
  on GPU it is now the dominant cost by a wide margin (21.5x eager slowdown vs 1.19x
  graph slowdown), and it requires no changes to this repo at all.
- It does **not** settle whether the §7 converter fix is still worth doing on GPU. At
  1.19x, the remaining CPU-motivated case for §7 (layout propagation) is much weaker for
  `v7_5_raw` specifically; whether that generalises depends on the untested models,
  especially `x3d_s` (Conv3D, the worst CPU offender at 40.6x) and the transformer-heavy
  rows where defect B already dominates on CPU.
