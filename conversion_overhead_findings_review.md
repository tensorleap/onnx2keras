# Critical review of `conversion_overhead_findings.md`

Reviewer pass over the findings doc. Every claim below is marked **validated** (checked
against source, or recomputed from the doc's own numbers) or **unvalidated** (reasoning
only, not measured).

Repo state at review time: `master` @ `b54841e`, working tree clean apart from `.DS_Store`
and the findings doc itself.

---

## 0. Overall verdict

The root-cause analysis is correct and the source citations are exact. Defect A
(transpose sandwich) is real, correctly localised, and correctly diagnosed as
non-optimisable downstream. The fix direction in §7 is the right one.

What does not hold up:

1. Defect B's causal story is contradicted by the doc's own survey data.
2. The headline "30% of compute" figure is less precise than stated, and the two methods
   presented as agreeing do not agree.
3. The evidence base is ~half the local corpus, and the benchmark ran a non-default
   production setting. Nothing is reproducible — no scripts committed.

None of this overturns the recommendation to fix Defect A. It does mean §7.3's predicted
outcome and §8's prioritisation are not yet on solid ground.

---

## 1. What checks out (validated)

All source citations are exact:

| Doc claim | Status |
|---|---|
| `keras_data_format_converter/layers/layer_utils.py:14` — `onnx_channel_first_cant_run_on_cpu_layers = (Conv, Pooling1D, Pooling2D, Pooling3D)` | verbatim match |
| `layer_utils.py:30-31` — `isinstance(...)` → `handle_data_format('channels_last', config)` | correct lines, correct code |
| `modelconverter.py:172-190` — the Permute-wrap `elif` block | exact; `elif` is line 172, `converted_tensor = permute_after` is line 190. Quote elides only the dynamic-channel `build` block at 177-182 |
| `pyproject.toml:19` — `keras-data-format-converter = "0.1.24"` | correct |
| §8.4 bug on branch `fix-bug-reshap-layers` — `K.is_keras_tensor(inp)` called before `tf.is_tensor(inp)` | confirmed on that branch; `master` genuinely does not contain this code. Reorder fix is right |

All arithmetic recomputes correctly: §4.2 layer sums (234, 350, +116), §4.2 permutation
pattern counts (67+66+3=136), every Perm/Conv ratio and every slowdown ratio in §5, the
§4.6 share denominator (components sum to exactly 111.34), and 111.34 + 126.88 = 238.22
matching the §3.4 thread-time caveat. The document is numerically self-consistent.

---

## 2. Finding 1 — Defect B's evidence contradicts the doc's own footnote and data

**Severity: high (invalidates a section's causal claim and §8.1's headline number).
Validated — recomputed from the §5 table.**

### 2a. The disqualified rows become the lead evidence

§5 footnote †, applied to `mnist-12` and `infineon`:

> † sub-millisecond absolute; ratio is dominated by fixed overhead, **ignore**.

§5.2 then builds its dispatch table on those rows and writes:

> `infineon` is the clean isolation of defect B: only 10 Permutes, but 72 lambdas from 96
> ONNX nodes, and a 37× eager penalty

`infineon`'s graph time is 0.21 ms. It is the row the doc just told the reader to ignore.

### 2b. Lambda count does not predict dispatch penalty

Eager/graph ratio computed for **all** survey rows, not the 6 selected:

| lambdas | model | dispatch penalty | in §5.2 table? |
|---|---|---|---|
| 3992 | swin_v2_t | **1.35×** | no |
| 1225 | rtdetrv2 | 2.44× | yes |
| 559 | traffic_light | **1.18×** | no |
| 407 | dinov2 | 2.35× | yes |
| 363 | ctformer | 7.8× | yes |
| 101 | kiwibot | 1.63× | no |
| 72 | infineon | 37× | yes |
| 3 | mnist-12 | 8.6× | yes |

The two models with the most lambdas have the smallest penalties, and both are omitted
from the table. The asserted correlation is absent — arguably negative.

### 2c. What the data actually supports

`(eager − graph) / layer_count` is roughly flat across model families:

| model | µs/layer |
|---|---|
| infineon | 67 |
| mnist-12 | 76 |
| dinov2 | 110 |
| rtdetrv2 | 112 |
| v7_5_raw | 135 |
| kiwibot | 140 |
| x3d_s | 140 |
| yolov7-tiny | 152 |
| yolo11s | 234 |
| traffic_light | 259 |
| ctformer | 295 |
| swin_v2_t | 511 |

(`chip` is an outlier at ~1230 µs/layer, consistent with its mis-resolved batch dimension
per §3.5.)

The penalty is per-Keras-layer Python dispatch at ~0.1–0.3 ms/layer. The *ratio* only
measures how little real compute a model has.

### 2d. Consequence for §8.1

> Wrapping the loaded model in `tf.function` at inference time ... would recover
> **2×–37×** on the transformer models for free ... probably the single highest
> value-per-effort item

The 37× is `infineon`, a Conv1D model where the absolute saving is 7.6 ms. On the actual
transformers the recovery is 1.35× (swin) and 2.44× (rtdetrv2). The absolute case is
still strong — swin saves ~2.2 s/call — but it must be argued in milliseconds, not in a
ratio range borrowed from the two disqualified rows.

**Recommended edit:** drop the ratio framing from §5.2 and §8.1; replace with per-layer
dispatch cost plus absolute savings; remove `infineon`/`mnist-12` from the dispatch table
or carry the † caveat with them.

---

## 3. Finding 2 — "two independent methods, which agree" (§3.4 / §4.6 / §4.7)

**Severity: high (the 30% figure is the entire justification for the §7 fix).
Validated — recomputed from the doc's own numbers.**

### 3a. Method 1 uses the wrong denominator

> Result for `v7_5_raw`: **26.1 ms** across 136 Permutes, against a 94.7 ms full-model
> eager time → ≈28%.

Numerator is graph-mode standalone wall time; denominator is *eager* wall time, which
§4.4 itself says contains ~47 ms of pure Python dispatch. Against the quantity the fix
would actually reduce (graph time, 50.9 ms), Method 1 gives **26.1 / 50.9 = 51%**, not
28%. Method 2 gives 30%. The methods differ by ~1.7×; the "agreement" comes from
dividing one of them by a denominator ~1.9× too large.

### 3b. Attributions sum to more than the measured runtime

Held to the eager baseline consistently:

| source | component | ms |
|---|---|---|
| §4.7 | ZeroPadding2D + LeakyReLU + Conv2D | 44.0 |
| §3.4 Method 1 | 136 Permutes | 26.1 |
| §4.4 | Python dispatch (98.2 − 50.9) | 47.3 |
| | **total** | **117.4** |
| | measured eager | **98.2** |

120% of runtime, before MaxPool, ConcatV2, BiasAdd and 57 lambdas. At least one
ingredient is inflated by ≥20%. Most likely cause (**unvalidated**): standalone re-timing
pays full call-entry and cold-cache cost per op and loses in-graph locality, so it
over-attributes. §4.7 has the same defect.

### 3c. A thread-time share is silently reused as a wall-time prediction

§3.4 states the caveat correctly:

> These durations are **CPU time summed across worker threads**, not wall time ...
> valid as *relative shares*, and not as absolute latencies.

§7.3 then converts it:

> removing substantially all transposes should take ~30% off graph-mode wall time,
> **51 → ~36 ms**

Valid only if `Transpose` and `_FusedConv2D` parallelise equally well. They almost
certainly do not — a large NCHW↔NHWC flip is memory-bandwidth-bound; `_FusedConv2D` is
heavily intra-op parallel.

Bounding from the doc's numbers: 111.34 ms op thread-time over ~51 ms wall ⇒ average
parallelism 2.18×. If `Transpose` parallelises at the average, its wall share is 30%. If
it is effectively serial, 33.84 ms of thread-time is 33.84 ms of **wall** time = **66% of
graph-mode runtime**.

True interval ≈ **30%–66%, with the doc's 30.4% sitting at the optimistic end.** The error
is directional and favours the fix, but "30.4%" to three significant figures and the
derived "51 → ~36 ms" overstate the measurement.

**Recommended action (~10 min, high value):** re-run the §3.4 profiler with
`tf.config.threading.set_intra_op_parallelism_threads(1)`. At one thread thread-time ≡
wall-time and the share becomes directly interpretable, collapsing the interval to a
single defensible number — before anyone spends a release cycle on
`keras-data-format-converter`.

### 3d. Unexplained op-count gap in the "authoritative" method

§4.6 reports `Transpose` count/iter = **115** and `Pad` = **21**, but §4.2 counts 136
`Permute` and 23 `ZeroPadding2D`, and §4.5 says:

> Frozen static-shape graph — still 136 `Transpose`. **All of them survive.**

21 transposes and 2 pads unaccounted for, unremarked. Either they are folded on
constant/weight paths at execution time — in which case "136 Permutes" overstates the
runtime problem by ~15% — or the profiler is dropping events, in which case 30.4% is
understated. Needs one sentence either way.

### 3e. §4.7 cross-checks ordering only

> Consistent with the profiler ordering.

Two methods both ranking conv first is near-zero information, and the section discloses it
never finished (`errored out before reaching Concatenate/MaxPooling2D`). Fine as a sanity
check; should not sit under a heading implying quantitative corroboration.

---

## 4. Finding 3 — scope, selection and reproducibility

**Severity: medium-high (affects the regression plan in §7.4). Validated by
filesystem inspection.**

### 4a. "the 13 ONNX models available locally in this repo" is inaccurate

The repo holds **22 unique ONNX models** (29 files, some duplicated between `./` and
`./test/`). Twelve local models were surveyed, plus the external `v7_5_raw`.

Unsurveyed: `maskrcnn`, `nms_v2`, `rfdetr-base`, `raft`, `clip`, `interfuser_planKD`,
`mmdet_convnext`, `split_model`, `asensus/lung_anatomy_merged`,
`asensus/lung_anatomy_merged_ir9`, `asensus/TrackerInferenceFcudarc3aug`.

Roughly half the corpus, and the omissions are not random: `maskrcnn`, `nms_v2`, `raft`,
`split_model` are the control-flow / NMS / dynamic-shape graphs where the §7.1 layout
rewrite is most likely to break. §7.4 leans on the survey as pre-built coverage:

> the 13 models in §5 make a ready-made regression set

If those models were excluded because they fail to convert on `master`, that is material
to the plan. If excluded for time, say so.

Loose end: `test/model.onnx` (BERT-like) supplies a row in §5.2's dispatch table
(84.0 / 8.6 ms) but never appears in the §5 survey table — a 14th, half-included model.

### 4b. The benchmark did not run the production default

§3.1 asserts:

> The pipeline is exactly what `onnx2kerastl/convert_model.py:8-26` does in production

but benchmarks `should_transform_inputs_and_outputs=False`. Actual code:

```python
def convert_onnx_to_keras(onnx_model_path, transform_io:bool = True):
    ...
    final_model = convert_channels_first_to_last(keras_model,
                                                 should_transform_inputs_and_outputs=transform_io,
                                                 verbose=True)
```

Default is `True`. Every number in the document is from the non-default path. The flag
governs boundary transposes — exactly the region §7.1 finds hardest — and §7.2 commits to
preserving both settings, so the plan promises to preserve behaviour on a setting never
baselined. "i.e. what actually ships" needs verifying against how Tensorleap actually
invokes the converter.

Two smaller inaccuracies in the same snippet: the comment `# minus initializers` describes
filtering the real code does not perform, and `verbose=True` is dropped.

### 4c. h5 round-trip not stated

§3.2 times whatever object conversion returned. Production saves an h5 and §8.1 says
Tensorleap runs the h5 eagerly. These need not be the same graph — `layer_utils.py:24`
shows revived layers take a distinct path:

```python
if isinstance(current_layer, RevivedLayer):
    logger.debug(f"Layer skipped, ...")
```

If the in-memory model was measured, the numbers describe an artifact that is not the one
served. One clarifying sentence closes this. (**Unvalidated** whether the h5 differs in
practice.)

### 4d. Nothing is reproducible

No measurement script exists in the tree — the only file matching
`*bench*`/`*profil*`/`*overhead*` is the findings doc. Embedded fragments are elided
(`... time f(src) over 20 iterations`) and cannot be run. Iteration count `n` is defined
in §3.2 but never given for the headline §4.4 or §5 numbers (Method 1 says 20, profiler
says 10, tables say nothing). Against the disclosed ±10–20% variance, `n` is load-bearing.
For a document requesting a regression suite and a cross-repo release, the harness *is*
most of that suite and should be committed.

### 4e. One overclaim ruled out by the doc's own variance caveat

§5.1:

> `yolov7-tiny` is effectively a clone of your model's profile — **135 vs 136 Permutes,
> 58 vs 58 convs, 2.57× vs 2.53×**

Layer counts are deterministic and convincing. `2.57× vs 2.53×` is not: §2 discloses
±10–20% variance, 4–8× larger than the 1.6% difference presented as agreement. Drop the
timing pair; the structural counts carry the argument alone.

---

## 5. Not yet delivered

Reviewed but not yet written up in detail (raised here so the list is complete):

1. **§7.1 bucket taxonomy has an unhandled middle category.** `Shape`/`Gather`/
   `Unsqueeze` lambdas are neither "elementwise TFOpLambda" (bucket 1) nor "TFOpLambda
   with baked-in axes" (bucket 3). Under layout propagation a `Shape` op would read NHWC
   dims where the detect head expects NCHW — a silent numerical corruption rather than a
   crash. `v7_5_raw` has 9 such `Shape` lambdas feeding `model.77`. **Unvalidated** —
   reasoning from §4.3's listing, not from a run.

2. **§1 and §8.3 attribute `x3d_s`'s 40.6× to Defect A without measuring it.** No
   profiler run was done on `x3d_s`; §5.1 argues from `eager ≈ graph` that it is memory
   traffic, which is suggestive but not attribution — TF's Conv3D kernels are
   independently far slower than ORT's. If most of the 40.6× is kernel quality, layout
   propagation will not fix it and the §8.3 prioritisation is wrong. **Unvalidated.**

3. **Product-visible side effect not covered by §7.4.** Removing ~116 `Permute` layers
   changes the layer graph Tensorleap displays and maps analyses onto. §7.4 gates on
   numerics and transpose counts only, not on downstream consumers of layer structure or
   naming (`name_policy='attach_weights_name'`). **Unvalidated.**

---

## 6. Suggested order of work

1. Single-thread profiler re-run to pin the transpose share (§3c) — ~10 min, gates the
   whole §7 justification.
2. Establish why 10 local models were excluded; if they fail to convert, record it (§4a).
3. Baseline `should_transform_inputs_and_outputs=True` (§4b).
4. Commit the measurement harness (§4d).
5. Correct §5.2 / §8.1 to absolute savings (§2d).
6. Only then decide between §7 (Defect A, cross-repo release) and §8.1
   (`tf.function` wrapping in engine/code-loader, no conversion changes).
