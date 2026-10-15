"""Tests for ggufcheck. Run with: python3 -m pytest tests/ (or python3 tests/test_ggufcheck.py)"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent.parent / "ggufcheck.py")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ggufcheck as gc  # noqa: E402


# ---------------------------------------------------------------------------
# a tiny GGUF writer so tests control every byte
# ---------------------------------------------------------------------------

def _w_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _w_kv(k, t, v):
    out = _w_str(k) + struct.pack("<I", t)
    if t == 4:
        out += struct.pack("<I", v)
    elif t == 7:
        out += struct.pack("<B", int(v))
    elif t == 8:
        out += _w_str(v)
    elif t == 9:
        et = 4
        out += struct.pack("<I", et) + struct.pack("<Q", len(v))
        out += b"".join(struct.pack("<I", x) for x in v)
    return out


def build_gguf(path: Path, arch="testarch", n_layer=4, *, interval=1,
               head_kv=2, head_size=64, embd=256, ctx=4096, ssm=False,
               layer_sizes=None, outer_size=1024, align=32):
    """layer_sizes: bytes per block tensor (f16 => elems = bytes/2)."""
    if layer_sizes is None:
        layer_sizes = [2048 * (i + 1) for i in range(n_layer)]  # increasing
    metas = [
        _w_kv("general.architecture", 8, arch),
        _w_kv("general.name", 8, "Test Model"),
        _w_kv(f"{arch}.block_count", 4, n_layer),
        _w_kv(f"{arch}.embedding_length", 4, embd),
        _w_kv(f"{arch}.attention.head_count", 4, 4),
        _w_kv(f"{arch}.attention.head_count_kv", 4, head_kv),
        _w_kv(f"{arch}.attention.key_length", 4, head_size),
        _w_kv(f"{arch}.context_length", 4, ctx),
        _w_kv("general.alignment", 4, align),
    ]
    if interval > 1:
        metas.append(_w_kv(f"{arch}.full_attention_interval", 4, interval))
    if ssm:
        metas.append(_w_kv(f"{arch}.ssm.inner_size", 4, 512))
        metas.append(_w_kv(f"{arch}.ssm.state_size", 4, 128))
        metas.append(_w_kv(f"{arch}.ssm.conv_kernel", 4, 4))

    tensors = []  # (name, nelements, type)
    off = 0
    for i, size in enumerate(layer_sizes):
        tensors.append((f"blk.{i}.ffn.weight", size // 2, 1))  # f16
        off += size
    tensors.append(("token_embd.weight", outer_size // 2, 1))
    off += outer_size

    hdr = b"GGUF" + struct.pack("<I", 3)
    hdr += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metas))
    hdr += b"".join(metas)
    cur = 0
    infos = b""
    data = b""
    for name, ne, ty in tensors:
        infos += _w_str(name) + struct.pack("<I", 1)
        infos += struct.pack("<Q", ne) + struct.pack("<I", ty)
        infos += struct.pack("<Q", cur)
        nbytes = ne * 2  # f16
        data += b"\x00" * nbytes
        cur += nbytes
    body = hdr + infos
    pad = (-len(body)) % align
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + b"\x00" * pad + data)
    return path


def run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv],
                          capture_output=True, text=True)


# ---------------------------------------------------------------------------

def test_parses_header_and_weights(tmp_path):
    sizes = [1024, 2048, 4096, 8192]
    p = build_gguf(tmp_path / "m.gguf", layer_sizes=sizes, outer_size=512)
    m = gc.Model(p)
    assert m.arch == "testarch" and m.n_layer == 4
    assert m.layer_bytes == {0: 1024, 1: 2048, 2: 4096, 3: 8192}
    assert m.outer_bytes == 512
    assert m.weights == sum(sizes) + 512
    assert m.quant == "f16"


def test_prefix_weights_matches_llama_semantics(tmp_path):
    sizes = [1024, 2048, 4096, 8192]
    p = build_gguf(tmp_path / "m.gguf", layer_sizes=sizes, outer_size=512)
    m = gc.Model(p)
    assert m.prefix_weights(0) == 0
    assert m.prefix_weights(2) == 1024 + 2048          # no embeds at partial
    assert m.prefix_weights(4) == sum(sizes) + 512     # embeds at full


def test_kv_math_plain_and_hybrid(tmp_path):
    plain = build_gguf(tmp_path / "plain.gguf", n_layer=4)
    m = gc.Model(plain)
    assert m.kv_layers() == 4
    # 2 (K+V) * 4 layers * 2 kv heads * 64 * 2 bytes f16 = 2048/token
    assert m.kv_bytes_per_token("f16") == 2048
    assert m.kv_bytes_per_token("q8_0") == 1024

    hyb = build_gguf(tmp_path / "hyb.gguf", n_layer=8, interval=4)
    h = gc.Model(hyb)
    assert h.kv_layers() == 2
    assert h.kv_bytes_per_token("f16") == 2 * 2 * 2 * 64 * 2


def test_plan_full_fit_and_partial_offload(tmp_path):
    sizes = [100_000] * 10
    p = build_gguf(tmp_path / "m.gguf", n_layer=10,
                   layer_sizes=sizes, outer_size=1000)
    m = gc.Model(p)
    kv_total = int(m.kv_bytes_per_token("f16") * 1000)
    # budget = exactly 7 layers + KV + overhead: 8 layers would not fit
    vram = 7 * 100_000 + kv_total + 500_000
    pl = gc.plan(m, 1000, "f16", vram, 10**12, 500_000)
    assert pl["max_ngl"] == 7
    assert not pl["fits_full_gpu"]
    assert pl["ram_at_max_ngl"] == 3 * 100_000 + 1000   # 3 layers + embeds
    # generous budget: everything fits
    pl2 = gc.plan(m, 1000, "f16", 10**10, 10**12, 500_000)
    assert pl2["fits_full_gpu"] and pl2["max_ngl"] == 10
    assert pl2["max_ctx_full_gpu"] > 1000


def test_cli_json(tmp_path):
    p = build_gguf(tmp_path / "m.gguf")
    r = run("--json", "--vram", "12", str(p))
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["models"][0]["arch"] == "testarch"
    assert d["models"][0]["plan"]["max_ngl"] >= 0


def test_cli_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" * 10)
    r = run(str(bad))
    assert r.returncode == 2
    assert "not a GGUF" in r.stderr


def test_cli_directory_mode_table(tmp_path):
    build_gguf(tmp_path / "a.gguf")
    build_gguf(tmp_path / "sub" / "b.gguf")
    (tmp_path / "ignore.txt").write_text("hi")
    r = run("--vram", "12", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("Test Model") == 2
    assert "ngl-fit" in r.stdout


def test_ssm_state_counted(tmp_path):
    p = build_gguf(tmp_path / "hyb.gguf", n_layer=8, interval=4, ssm=True)
    m = gc.Model(p)
    assert m.ssm
    ssm_bytes = m.ssm_bytes_per_seq()
    assert ssm_bytes == 8 * 512 * (128 + 4) * 4
    pl = gc.plan(m, 100, "f16", 10**10, 10**12, 0)
    assert pl["kv_total"] == int(m.kv_bytes_per_token("f16") * 100 + ssm_bytes)


if __name__ == "__main__":
    import inspect
    import tempfile
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                try:
                    fn(Path(d))
                    print(f"  ok  {name}")
                except Exception as e:
                    failed += 1
                    print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
