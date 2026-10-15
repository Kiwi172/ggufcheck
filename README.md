# ggufcheck

[![tests](https://github.com/Kiwi172/ggufcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/Kiwi172/ggufcheck/actions/workflows/ci.yml)

**Will this GGUF fit — and at how much context? Answered from the file
header, before llama.cpp dies at 80% loaded.**

You downloaded a 21 GB Q4_K_M for a 12 GB card. llama.cpp will happily
begin loading it, churn for two minutes, and then fail on a `cudaMalloc` —
and `llama-bench` passing at short context does *not* prove your 32k
context fits, because the KV cache is allocated separately. The arithmetic
to answer this lives in the GGUF header (exact per-layer sizes, KV
geometry, hybrid-SSM attention intervals) and in your machine. `ggufcheck`
reads both and does the math in a tenth of a second.

```
   the GGUF file                          your machine
   ┌──────────────────────┐              ┌──────────────────┐
   │ per-tensor offsets   │              │ /proc/meminfo    │
   │  → exact layer sizes │              │ nvidia-smi       │
   │ n_layer, kv heads,   │   ggufcheck  │                  │
   │ head size, interval  │ ◀──────────▶ │                  │
   │ trained ctx, quants  │   (reads,    └──────────────────┘
   └──────────────────────┘    never
                  │            loads weights)
                  ▼
     full offload? NO — needs 22.1 GB, have 12.9
     largest -ngl that fits at ctx 32768: 23/40
     largest ctx at -ngl 23: 53,337 tokens
     suggested: llama-server -m model.gguf -ngl 23 -ctk q8_0 -ctv q8_0 -c 32768
```

It never loads the model — it reads the header and the filesystem metadata.
Checking a directory of 60 GB of GGUFs takes about a second. No network,
no llama.cpp required, no Python packages: one file, standard library only.

---

## Install

Any machine with Python 3.9 or newer:

```bash
curl -sO https://raw.githubusercontent.com/Kiwi172/ggufcheck/main/ggufcheck.py
chmod +x ggufcheck.py
```

Or:

```bash
pipx install git+https://github.com/Kiwi172/ggufcheck
```

## Use

One model, the full verdict:

```
$ ggufcheck ~/models/Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf --ctx 32768 --kv q8_0
Huihui Qwen3.6 35B A3B Abliterated  ·  qwen35moe  ·  dominant quant q4_k
40 layers, 2048 wide, kv heads 2×256, hybrid SSM: KV only on 10/40 layers
weights 21.2 GB  (40 blocks + 703 MB embed/output)  ·  trained ctx 262144

budget: 12.9 GB VRAM (NVIDIA GeForce RTX 3060)
        71.4 GB RAM available

KV cache: 10 kB/token (q8_0) → 422 MB at ctx 32768 + 87 MB SSM state

  full GPU offload: NO  needs 22.1 GB, have 12.9 GB
  largest -ngl that fits at ctx 32768: 23 of 40 (GPU 12.6 GB, RAM 9.4 GB)
  full offload never fits at any ctx (weights alone exceed VRAM)
  largest ctx at -ngl 23: 53,337 tokens

  suggested: llama-server -m ".../Q4_K_M.gguf" -ngl 23 -ctk q8_0 -ctv q8_0 -c 32768 --flash-attn
```

A whole folder, comparison table:

```
$ ggufcheck ~/models/ --ctx 32768
model                                        quant      weights  ngl-fit  max-ctx@ngl
---------------------------------------------------------------------------
Huihui Qwen3.6 35B A3B Abliterated           q4_k       21.2 GB    22/40       53,512
Huihui Qwen3.6 35B A3B Abliterated           iq3_xxs    13.6 GB    35/40       41,771
Qwen3.8-27B                                  q4_k       16.5 GB    45/65       35,640
Huihui Gpt Oss 20b Mxfp4 Abliterated v2      mxfp4      13.8 GB    22/24       37,695
```

Plan for a *different* machine without being on it:

```bash
ggufcheck ~/models/ --vram 24 --ctx 131072   # a 4090, long context
```

## The math it does for you

- **Exact weight sizes per layer**, from tensor data offsets — not from
  quant-format tables, so it is correct even for formats older than this
  script (MXFP4, NVFP4, whatever lands next).
- **KV cache**: `2 × kv_layers × kv_heads × head_size × bytes_per_elem × ctx`,
  with `--kv f16|q8_0|q4_0` matching llama.cpp's `-ctk/-ctv`.
- **Hybrid models**: Qwen3-Next-style SSM/MoE architectures only keep a
  KV cache on every `full_attention_interval`-th layer — ggufcheck counts
  those correctly and adds the per-sequence recurrent state estimate.
  (This is why a 40-layer hybrid can hold 50k context on a 12 GB card
  when a dense model can't hold 8k.)
- **Offload simulation**: `prefix_weights(ngl)` mirrors llama.cpp's rule
  that embeddings/output join the GPU only at full offload, then binary
  search for the largest `-ngl` and the largest context that fit.

## Flags

| flag | meaning |
|---|---|
| `--ctx N` | context length to plan for (default 8192) |
| `--kv f16\|q8_0\|q4_0` | KV quantization, matches `-ctk/-ctv` |
| `--vram GB` / `--ram GB` | plan for another machine |
| `--overhead-mb N` | compute-buffer headroom (default 500) |
| `--json` | machine-readable |

## Non-goals

- **Not a benchmark.** It tells you what fits, not how fast it runs.
  MoE offload tricks (`--n-cpu-moe`) change speed, not fit — the fit
  numbers hold either way.
- **Estimates are honest estimates.** The compute-buffer overhead default
  (500 MB) and the SSM-state term are approximations; the weight and KV
  numbers are exact. When ggufcheck says "fits", `llama-bench` is still
  the proof.

## Tests

```bash
python3 -m pytest tests/      # or: python3 tests/test_ggufcheck.py
```

The tests build synthetic GGUF files byte by byte, so every size, offset
and verdict is checked against arithmetic the test itself controls.

## License

MIT
