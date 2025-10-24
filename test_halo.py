# test_helo.py — Validierung & Benchmark für HALO v0.3
from __future__ import annotations
from array import array
from time import perf_counter
from pathlib import Path
import json
from halo import HALO

def check_correctness(n: int = 1_000_000) -> None:
    """Einfacher Korrektheitscheck: y = 1 + 0.25*0.5 = 1.125 ⇒ Summe = 1.125 * n"""
    x = array('f', [0.5]*n)
    y = array('f', [1.0]*n)
    h = HALO(n_autotune=min(n, 1_000_000), iters=5, enable_streaming=True, stream_threshold=500_000, threads=1)
    h.saxpy(0.25, x, y)
    s = h.sum(y)
    exp = (1.0 + 0.25*0.5) * n
    print(f"[CHECK] sum(y) = {s:.1f} | expected = {exp:.1f}")
    assert abs(s - exp) < 1e-2

    print("Features:", h.features)
    prof = json.loads((Path.home()/".halo_profile.json").read_text(encoding="utf-8"))
    print("Profil:", prof)

def bench(label: str, fn, repeats=5) -> float:
    t0 = perf_counter()
    for _ in range(repeats):
        fn()
    dt = (perf_counter() - t0) / repeats
    return dt

def run_bench(n: int, threads: int) -> None:
    x = array('f', [0.5]*n)
    y = array('f', [1.0]*n)

    # Streaming aktiv, Schwellwert moderat, Threads konfigurierbar
    h = HALO(n_autotune=min(n, 1_000_000), iters=5,
             enable_streaming=True, stream_threshold=500_000, threads=threads)

    # Warmup (Caches/Branch-Predictor)
    h.saxpy(0.25, x, y)

    # Single-Thread
    dt_st = bench("ST", lambda: h.saxpy(0.25, x, y))
    gbps_st = (8 * n) / dt_st / 1e9
    print(f"[ST ] n={n:,} | {dt_st*1e3:7.3f} ms | ~{gbps_st:6.2f} GB/s")

    # Multi-Thread
    dt_mt = bench("MT", lambda: h.saxpy_mt(0.25, x, y))
    gbps_mt = (8 * n) / dt_mt / 1e9
    print(f"[MT x{threads}] n={n:,} | {dt_mt*1e3:7.3f} ms | ~{gbps_mt:6.2f} GB/s")

if __name__ == "__main__":
    check_correctness(1_000_000)
    for n in (1_000_000, 5_000_000, 20_000_000):
        run_bench(n, threads=4)  # für SAXPY meist ohne Vorteil; hier nur zum Vergleich
