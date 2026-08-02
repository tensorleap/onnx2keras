# Conversion overhead benchmarks

Tooling behind [`../conversion_overhead_findings.md`](../conversion_overhead_findings.md).
Measures what the ONNX → Keras conversion pipeline costs at inference time, relative to
running the same ONNX graph under onnxruntime.

All three tools run the production pipeline from `onnx2kerastl/convert_model.py`:
`onnx_to_keras(...)` followed by `convert_channels_first_to_last(...)`. The intermediate
channels-first model cannot run on CPU at all, so only the final model is measured.

## Tools

| script | answers |
|---|---|
| `survey.py` | How many layers/Permutes/lambdas does each model produce, and how much slower is it than onnxruntime? |
| `profile_ops.py` | Within one model, which op types actually consume the time? |
| `layer_costs.py` | What does each layer type cost standalone, and how much data do the Permutes move? |

## Usage

```bash
# Single model
poetry run python -m benchmarks.survey test/models/yolo_v7/yolov7-tiny.onnx

# Full local corpus, appending JSONL
poetry run python -m benchmarks.survey --keep-going --out results.jsonl \
    test/models/mnist/mnist-12.onnx \
    test/models/yolo_v7/yolov7-tiny.onnx \
    test/yolov11/yolo11s.onnx \
    test/kiwibot/model.onnx \
    test/chip/All_3ChipTypes_seg_model_deployed.onnx \
    test/ctformer/ctformer.onnx \
    test/x3d_s.onnx \
    test/traffic_light/model.onnx \
    test/dinov2/dino-2-test.onnx \
    test/swin/swin_v2_t.onnx \
    test/rtdetrv2/rtdetrv2_r18vd_120e_raw_outputs.onnx \
    test/infineon/enhanced_trial_19_full_model_complete.onnx \
    test/model.onnx

# Per-op attribution for one model
poetry run python -m benchmarks.profile_ops model.onnx --logdir /tmp/prof

# Per-layer-type cost
poetry run python -m benchmarks.layer_costs model.onnx
poetry run python -m benchmarks.layer_costs model.onnx --types Permute Conv2D
```

## Dynamic input dimensions

Symbolic ONNX dims are resolved as: **axis 0 is always batch → 1**, regardless of its
name; dims named `h`/`height`/`w`/`width` → `--spatial` (default 640); dims named
`seq`/`sequence`/`tokens` → `--sequence` (default 128); anything else falls back to
`--spatial` for rank ≥ 4 inputs and `--sequence` otherwise.

The axis-0 rule exists because a name table cannot recognise every batch dim in the wild —
`swin_v2_t` uses `batch_size` and the chip segmentation model uses `unk__251`. An earlier
version of this harness defaulted unrecognised names to 224 and silently benchmarked both
models at **batch 224**, which is where the 4294 ms and 693 ms figures in the findings doc
came from. Those two rows are flagged in the doc; their ratios are valid because both
backends received the same tensor, but the absolute times are not representative.

Override explicitly when the defaults are wrong:

```bash
poetry run python -m benchmarks.survey model.onnx --dim height=1080 --dim width=1920
```

## Reading the numbers

**Four execution modes are timed and they differ a lot.** `model(x)` runs eagerly, so
Python dispatches every layer individually — with a few hundred layers that overhead can
dominate. `tf.function(...)` is the compiled graph and is the fastest the h5 can go.
`model.predict(x)` also compiles, but adds data-adapter, batching and callback plumbing;
it is built for looping over a dataset, not for one tensor. Quote graph mode when asking
"how fast can this model go", and eager when asking "how fast is it in the harness that
runs it today".

**Device synchronisation.** `common.bench` materialises outputs after every call
(`common.sync`). TensorFlow GPU ops are asynchronous, so without this a GPU benchmark
measures kernel-launch time rather than execution and reports absurdly fast results. Do
not remove it.

**Profiler durations are thread-time, not wall time.** `profile_ops.py` sums each op
type's duration across worker threads, so on a multi-core CPU the total can exceed wall
time. Use the *share* column, not the absolute ms. Runtime scope events (`<lambda>`,
`ExecutorState::Process`) enclose the kernels they dispatch and are excluded — including
them double-counts the whole forward pass. As a sanity check, the reported leaf-op sum
should land near wall time on a lightly loaded machine.

**`layer_costs.py` shares do not sum to 100%.** Each layer is re-timed standalone against
a full-model baseline that benefits from fusion (TF fuses `Conv2D`+`BiasAdd` into
`_FusedConv2D`), so the parts over-count relative to the whole. Use it for ordering
between layer types, not for a budget. It also uses real `tf.Variable` inputs rather than
`tf.zeros`, because TensorFlow constant-folds `tf.transpose(tf.zeros(...))` at trace time
and makes transposes appear free.

**Run-to-run variance is significant.** Repeat measurements on a loaded laptop varied by
±10–20%, and two profiler runs of `v7_5_raw` put `Transpose` at 30.4% and 22.4% of op
time. Ratios between backends measured in the same process are far more stable than
absolute times. Run on an idle machine and prefer medians over single runs.

## GPU

Nothing here is CPU-specific, and `describe_devices()` reports the TF devices and
onnxruntime providers actually in use. `ort_session` picks `CUDAExecutionProvider`
automatically when available; override with `--providers`.

Expect the conclusions to shift on GPU, in both directions:

- cuDNN natively prefers **NCHW**, the opposite of CPU, so TF may keep convolutions
  channels-first and insert fewer transposes — or none of the per-conv pattern at all.
- Transposes are bandwidth-bound, and GPU bandwidth is far higher, so their share should
  fall even where they remain.
- The eager-dispatch penalty should get **relatively worse**: kernels get much faster
  while Python dispatch cost is unchanged.

So defect A (transpose sandwich) likely shrinks on GPU while defect B (lambda explosion
and eager dispatch) likely grows. Since GPU is Tensorleap's usual deployment target, the
GPU numbers should be treated as the ones that set fix priority.

## Caveat: `--keep-going`

`survey.py` catches per-model conversion failures **only** under `--keep-going`, so one
bad model does not abort a long corpus run. The error is recorded in the JSONL record and
printed in the summary rather than swallowed. Without the flag, failures propagate.
