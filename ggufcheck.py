#!/usr/bin/env python3
"""ggufcheck -- will this GGUF fit, and at how much context?

You downloaded a 21 GB Q4_K_M. llama.cpp will happily start loading it and
then die on a cudaMalloc. ggufcheck reads the GGUF header itself -- exact
per-layer weight sizes, KV-cache geometry, hybrid-SSM attention intervals --
and answers before you wait: does it fit, what is the largest -ngl that
fits, what is the largest context that fits, and the exact llama-server
flags to run. Pure Python stdlib. No model download, no network, no
llama.cpp required.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# GGUF header parsing
# ---------------------------------------------------------------------------

GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4",
    40: "nvfp4", 41: "q1_0", 42: "q2_0",
}

_KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

BLK_RE = re.compile(r"^blk\.(\d+)\.")


class GgufError(Exception):
    pass


def _rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 28:
        raise GgufError("implausible string length; not a GGUF?")
    return f.read(n).decode("utf-8", "replace")


def _rd_val(f, t):
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
    if t == 8:
        return _rd_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_rd_val(f, et) for _ in range(n)]
    if t not in FMT:
        raise GgufError(f"unknown GGUF value type {t}")
    fmt = FMT[t]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


class Model:
    """Everything ggufcheck needs, read from one GGUF header."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise GgufError("not a GGUF file")
            (self.version,) = struct.unpack("<I", f.read(4))
            if self.version > 3:
                raise GgufError(f"GGUF version {self.version} unsupported")
            (self.n_tensors,) = struct.unpack("<Q", f.read(8))
            (n_kv,) = struct.unpack("<Q", f.read(8))
            self.meta = {}
            for _ in range(n_kv):
                k = _rd_str(f)
                (t,) = struct.unpack("<I", f.read(4))
                self.meta[k] = _rd_val(f, t)
            infos = []
            for _ in range(self.n_tensors):
                name = _rd_str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
                (ty,) = struct.unpack("<I", f.read(4))
                (off,) = struct.unpack("<Q", f.read(8))
                infos.append((name, ty, off))
            hdr_end = f.tell()
        # exact per-tensor sizes from data-section offsets: robust for every
        # quant format, including ones older than this script
        infos.sort(key=lambda x: x[2])
        fsize = path.stat().st_size
        align = int(self.meta.get("general.alignment", 32))
        data_start = (hdr_end + align - 1) // align * align
        self.layer_bytes = {}          # layer index -> weight bytes
        self.outer_bytes = 0          # embeddings, output, norms...
        self.weights = 0
        type_bytes = {}               # quant label -> bytes (for the label)
        for i, (name, ty, off) in enumerate(infos):
            # data section spans [data_start, EOF); tensor i's bytes run
            # from data_start+off to the next tensor's start (or EOF)
            end = infos[i + 1][2] if i + 1 < len(infos) else fsize - data_start
            size = end - off
            self.weights += size
            label = GGML_TYPE_NAMES.get(ty, f"type{ty}")
            type_bytes[label] = type_bytes.get(label, 0) + size
            m = BLK_RE.match(name)
            if m:
                self.layer_bytes[int(m.group(1))] = \
                    self.layer_bytes.get(int(m.group(1)), 0) + size
            else:
                self.outer_bytes += size

        a = self._arch()
        g = lambda key, default=None: self.meta.get(f"{a}.{key}", default)
        self.arch = a
        self.name = self.meta.get("general.name", path.stem)
        self.n_layer = int(g("block_count") or 0)
        self.n_embd = int(g("embedding_length") or 0)
        self.head_kv = int(g("attention.head_count_kv")
                           or g("attention.head_count") or 0)
        self.head_size = int(g("attention.key_length")
                             or (self.n_embd // max(int(g("attention.head_count") or 1), 1)))
        self.ctx_train = int(g("context_length") or 0)
        self.interval = int(g("full_attention_interval") or 1)
        self.ssm = g("ssm.inner_size") is not None
        # quant label = whichever format holds the most bytes (attention
        # tensors are often a higher quant than the FFN bulk; count-based
        # labels get that wrong, byte-based does not)
        self.quant = max(type_bytes, key=type_bytes.get) if type_bytes else "?"

    def _arch(self):
        a = self.meta.get("general.architecture")
        if not a:
            raise GgufError("no general.architecture in metadata")
        return a

    @property
    def n_layer_sorted(self):
        return sorted(self.layer_bytes)

    def kv_layers(self) -> int:
        """Layers that actually hold a KV cache (hybrid SSM: only some)."""
        if self.interval <= 1:
            return self.n_layer
        return sum(1 for i in range(self.n_layer) if (i + 1) % self.interval == 0) \
            or self.n_layer

    def kv_bytes_per_token(self, kv_type: str) -> float:
        return 2.0 * self.kv_layers() * self.head_kv * self.head_size \
            * _KV_BYTES[kv_type]

    def ssm_bytes_per_seq(self) -> int:
        """Rough recurrent-state estimate for hybrid models, per sequence."""
        if not self.ssm:
            return 0
        inner = int(self.meta.get(f"{self.arch}.ssm.inner_size") or 0)
        state = int(self.meta.get(f"{self.arch}.ssm.state_size") or 0)
        conv = int(self.meta.get(f"{self.arch}.ssm.conv_kernel") or 0)
        return self.n_layer * inner * (state + conv) * 4  # fp32 state

    def prefix_weights(self, ngl: int) -> int:
        """Weight bytes for the first `ngl` blocks, plus embeddings/output
        when everything is offloaded (llama.cpp behaviour)."""
        layers = self.n_layer_sorted[:ngl]
        total = sum(self.layer_bytes[i] for i in layers)
        if ngl >= self.n_layer:
            total += self.outer_bytes
        return total


# ---------------------------------------------------------------------------
# system detection
# ---------------------------------------------------------------------------

def detect_system():
    ram = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    ram = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    vram, gpu_name = None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        first = out.strip().splitlines()[0]
        gpu_name, mb = [s.strip() for s in first.split(",")]
        vram = int(float(mb) * 1024 * 1024)
    except Exception:
        pass
    return {"ram": ram, "vram": vram, "gpu": gpu_name}


# ---------------------------------------------------------------------------
# fit math
# ---------------------------------------------------------------------------

def plan(model: Model, ctx: int, kv_type: str, vram: int, ram: int,
         overhead: int):
    """Compute the fit verdicts for one model."""
    kv_tok = model.kv_bytes_per_token(kv_type)
    kv_total = kv_tok * ctx + model.ssm_bytes_per_seq()

    def gpu_need(ngl):
        return model.prefix_weights(ngl) + kv_total + overhead

    full = model.weights + kv_total + overhead
    fits_full = vram is not None and full <= vram

    max_ngl = 0
    if vram is not None:
        for ngl in range(model.n_layer + 1):
            if gpu_need(ngl) <= vram:
                max_ngl = ngl
            else:
                break

    max_ctx = None
    if vram is not None and kv_tok > 0:
        budget = vram - model.weights - overhead
        if budget > 0:
            max_ctx = int(budget / kv_tok)

    # partial offload: weights that stay in RAM when running at max_ngl
    ram_need = model.weights - model.prefix_weights(max_ngl)
    # largest ctx that fits at the best-fitting -ngl (the practical number)
    max_ctx_at_ngl = None
    if vram is not None and kv_tok > 0:
        budget = vram - model.prefix_weights(max_ngl) - overhead \
            - model.ssm_bytes_per_seq()
        if budget > 0:
            max_ctx_at_ngl = int(budget / kv_tok)
    return {
        "weights": model.weights,
        "kv_total": int(kv_total),
        "kv_per_token": int(kv_tok),
        "full_gpu_total": full,
        "fits_full_gpu": bool(fits_full),
        "max_ngl": max_ngl,
        "ram_at_max_ngl": ram_need,
        "max_ctx_full_gpu": max_ctx,
        "max_ctx_at_max_ngl": max_ctx_at_ngl,
    }


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def human(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} kB"


def check_mark(ok):
    return "\033[32mOK\033[0m " if ok else "\033[31mNO\033[0m "


def report(model: Model, p: dict, args, sysinfo):
    m = model
    print(f"{m.name}  ·  {m.arch}  ·  dominant quant {m.quant}")
    geo = (f"{m.n_layer} layers, {m.n_embd} wide, kv heads {m.head_kv}×{m.head_size}")
    if m.interval > 1:
        geo += f", hybrid SSM: KV only on {m.kv_layers()}/{m.n_layer} layers"
    print(geo)
    print(f"weights {gb(m.weights)}  ({len(m.n_layer_sorted)} blocks + "
          f"{human(m.outer_bytes)} embed/output)  ·  trained ctx {m.ctx_train}")
    print()
    if args.vram:
        vram = int(args.vram * 1e9)
        print(f"budget: {gb(vram)} VRAM (override)")
    elif sysinfo["vram"]:
        print(f"budget: {gb(sysinfo['vram'])} VRAM ({sysinfo['gpu']})")
    else:
        print("budget: no GPU detected (nvidia-smi absent) -- CPU-only view")
    if sysinfo["ram"]:
        print(f"        {gb(sysinfo['ram'])} RAM available")
    print()

    kv = p["kv_per_token"]
    print(f"KV cache: {human(kv)}/token ({args.kv}) → "
          f"{human(p['kv_total'])} at ctx {args.ctx}"
          + (f" + {human(m.ssm_bytes_per_seq())} SSM state" if m.ssm else ""))
    print()

    if args.vram:
        vram = int(args.vram * 1e9)
    else:
        vram = sysinfo["vram"]
    if vram:
        if p["fits_full_gpu"]:
            print(f"  full GPU offload (-ngl {m.n_layer}, ctx {args.ctx}): "
                  f"{check_mark(True)} {gb(p['full_gpu_total'])} of {gb(vram)}")
        else:
            print(f"  full GPU offload: {check_mark(False)} needs "
                  f"{gb(p['full_gpu_total'])}, have {gb(vram)}")
            print(f"  largest -ngl that fits at ctx {args.ctx}: "
                  f"{p['max_ngl']} of {m.n_layer} "
                  f"(GPU {gb(m.prefix_weights(p['max_ngl']) + p['kv_total'] + args.overhead_mb * 1e6)}, "
                  f"RAM {gb(p['ram_at_max_ngl'])})")
        if p["max_ctx_full_gpu"] is not None:
            print(f"  largest ctx at full offload: {p['max_ctx_full_gpu']:,} tokens")
        elif not p["fits_full_gpu"]:
            print("  full offload never fits at any ctx (weights alone exceed VRAM)")
        if p["max_ctx_at_max_ngl"] is not None and not p["fits_full_gpu"]:
            print(f"  largest ctx at -ngl {p['max_ngl']}: "
                  f"{p['max_ctx_at_max_ngl']:,} tokens")
        # suggested command
        ngl = p["max_ngl"]
        flags = f"-ngl {ngl}"
        if args.kv != "f16":
            flags += f" -ctk {args.kv} -ctv {args.kv}"
        flags += f" -c {args.ctx}"
        print()
        print(f"  suggested: llama-server -m \"{m.path}\" {flags} --flash-attn")
    else:
        total = p["full_gpu_total"]
        if sysinfo["ram"] and total <= sysinfo["ram"]:
            print(f"  CPU inference: {check_mark(True)} {gb(total)} resident, "
                  f"fits in available RAM")
        else:
            print(f"  CPU inference: {check_mark(False)} needs {gb(total)}")


def table(models, args, sysinfo):
    vram = int(args.vram * 1e9) if args.vram else sysinfo["vram"]
    hdr = (f"{'model':<44} {'quant':<8} {'weights':>9} {'ngl-fit':>8} "
           f"{'max-ctx@ngl':>12}")
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        p = plan(m, args.ctx, args.kv, vram or 0, sysinfo["ram"] or 0,
                 args.overhead_mb * 1024 * 1024)
        name = m.name[:44]
        mc = (f"{p['max_ctx_at_max_ngl']:,}"
              if p["max_ctx_at_max_ngl"] else "-")
        print(f"{name:<44} {m.quant:<8} {gb(m.weights):>9} "
              f"{str(p['max_ngl']) + '/' + str(m.n_layer):>8} {mc:>12}")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="ggufcheck",
        description="Will this GGUF fit, and at how much context? Reads the "
                    "file header and your machine; prints the verdict and "
                    "llama-server flags. Stdlib-only, no network.")
    ap.add_argument("paths", nargs="+", metavar="PATH",
                    help="GGUF file(s) or directories containing them")
    ap.add_argument("--ctx", type=int, default=8192,
                    help="context length to plan for (default: 8192)")
    ap.add_argument("--kv", choices=list(_KV_BYTES), default="f16",
                    help="KV cache quantization (default: f16)")
    ap.add_argument("--vram", type=float, default=None, metavar="GB",
                    help="override VRAM budget in GB (for planning other boxes)")
    ap.add_argument("--ram", type=float, default=None, metavar="GB",
                    help="override available RAM in GB")
    ap.add_argument("--overhead-mb", type=int, default=500,
                    help="compute-buffer headroom in MB (default: 500)")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--version", action="version",
                    version=f"ggufcheck {VERSION}")
    return ap


def collect(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(p.rglob("*.gguf")))
        elif p.is_file():
            out.append(p)
        else:
            raise GgufError(f"no such path: {p}")
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        models = [Model(p) for p in collect(args.paths)]
    except GgufError as e:
        print(f"ggufcheck: {e}", file=sys.stderr)
        return 2
    if not models:
        print("ggufcheck: no .gguf files found", file=sys.stderr)
        return 2

    sysinfo = detect_system()
    if args.ram:
        sysinfo["ram"] = int(args.ram * 1e9)
    overhead = args.overhead_mb * 1024 * 1024
    vram = int(args.vram * 1e9) if args.vram else sysinfo["vram"]

    if args.as_json:
        print(json.dumps({
            "tool": "ggufcheck", "version": VERSION,
            "system": sysinfo, "ctx": args.ctx, "kv": args.kv,
            "models": [{"path": str(m.path), "name": m.name, "arch": m.arch,
                        "quant": m.quant, "weights": m.weights,
                        "n_layer": m.n_layer, "kv_layers": m.kv_layers(),
                        "plan": plan(m, args.ctx, args.kv, vram or 0,
                                     sysinfo["ram"] or 0, overhead)}
                       for m in models],
        }, indent=2))
        return 0

    if len(models) > 1:
        table(models, args, sysinfo)
    else:
        report(models[0], plan(models[0], args.ctx, args.kv, vram or 0,
                               sysinfo["ram"] or 0, overhead), args, sysinfo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
