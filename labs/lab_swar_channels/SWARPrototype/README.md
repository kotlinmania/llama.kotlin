# SWARPrototype

Swift sketch for the kcoro/SWAR tile executor. This binary mimics the CPU-side dequant stage by
streaming `block_q8_0` tiles (32×int8 + one fp16 scale) and converting them into `Float16` buffers.

## Why Swift?

kcoro’s C++ coroutine layer is still stabilizing. This prototype lets us iterate on the SWAR math
and async orchestration without waiting on the C extension:

- Uses Swift Structured Concurrency (`withTaskGroup`) to mirror the Kotlin coroutine chunking.
- Emulates the gs=32 SWAR kernel so we can benchmark from user space (`swift run ... --configuration release`).
- Provides a checksum to keep the optimizer honest while we profile.

Once the kcoro_py bridge lands, this code maps directly onto the planned coroutine job graph:
GGUF reader → SWAR dequant → MLX upload → recycler.

## Running

```bash
cd swift/SWARPrototype
swift run --configuration release swar-prototype -- --blocks 65536
```

Enable the Metal-backed path for a GPU baseline:

```bash
swift run --configuration release swar-prototype -- --blocks 65536 --metal
```

Run the Combine broadcast demo (multi-subscriber CPU pipeline):

```bash
swift run --configuration release swar-prototype -- --blocks 65536 --combine
```

Exercise the actor-based broadcast prototype (each coroutine acts like a "player piano" key):

```bash
swift run --configuration release swar-prototype -- --blocks 65536 --actor
```

The tool auto-scales iteration counts using the same heuristic as the Kotlin bench
(`<=64` blocks → `200_000` repeats, `<=4096` → `50_000`, otherwise `2_000`). Override with
`--iterations` when profiling specific workloads. Concurrency fan-out defaults to
`min(activeProcessors, 4)`; tweak it with `--workers` (logical threads) and `--chunks`
(number of coroutine slices per iteration). When `--metal` is present the CPU chunk fan-out is
bypassed and the summary line will report `engine=metal`; `--combine` swaps in the Combine
multi-subscriber path and prints `engine=combine`; `--actor` drives the actor-based broadcaster
and tags the summary with `engine=actor`.

### Using a GGUF model

Point the prototype at a real GGUF file and tensor to dequantize genuine Q8 tiles:

```bash
swift run --configuration release swar-prototype -- \
  --gguf /path/to/model.gguf --tensor blk.0.attn.wq --blocks 8192
```

If `--tensor` is omitted the tool picks the first `Q8_0` tensor it finds. Blocks default to 4096.

⚠️ Only `Q8_0` tensors are supported at the moment. Use `--tensor` to point at a compatible block
or re-run with synthetic data until the `Q8_K` path is wired in.

The executor now keeps everything in integer math: each tile returns `values` (Int32 numerators) plus
an exponent per block so `value * 2^exponent` recreates the dequantized float. Throughput is reported
as logical GB/s (raw quantized bytes) so we can compare directly with the coming kcoro/NEON helpers.

## Next steps

- Replace the `DispatchQueue` shim with kcoro once the extension is ready.
- Tune the Metal kernel (tiling, vectorized loads) and compare sustained GB/s against the CPU path.
- Feed real GGUF tiles by hooking this executable up to `gguf_direct.py` when the parser lands.
