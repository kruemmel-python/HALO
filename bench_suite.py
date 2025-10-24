#!/usr/bin/env python3
"""Benchmark-Suite für HALO-Operationen.

Erzeugt eine CSV-Tabelle mit Durchsatzmessungen für verschiedene Problemgrößen
und Thread-Konfigurationen.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from halo import (
    HALO,
    Impl,
    make_identity_lut,
    make_aligned_u8_buffer,
    make_aligned_f32_buffer,
    make_pinned_float_array,
)

@dataclass
class BenchResult:
    name: str
    width: int
    height: int
    threads: int
    elapsed: float
    notes: str = ""


def _timeit(fn, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def bench_saxpy(halo: HALO, n: int) -> float:
    x = make_pinned_float_array(n)
    y = make_pinned_float_array(n)
    def run():
        halo.saxpy(0.5, x, y)
    return _timeit(run)


def bench_sum(halo: HALO, n: int) -> float:
    x = make_pinned_float_array(n)
    def run():
        halo.sum(x)
    return _timeit(run)


def bench_img_u8(halo: HALO, width: int, height: int) -> float:
    buf, stride = make_aligned_u8_buffer(width, height)
    out, ostride = make_aligned_f32_buffer(width, height)
    lut = make_identity_lut()
    def run():
        halo.img_u8_to_f32_lut_axpby_2d(buf, width, height, stride, out, ostride, lut)
    return _timeit(run)


def bench_img_u16(halo: HALO, width: int, height: int, bit_depth: int = 10) -> float:
    stride = ((width * 2 + 63) // 64) * 64
    buf, _ = make_aligned_u8_buffer(width * 2, height, stride_bytes=stride)
    out, ostride = make_aligned_f32_buffer(width, height)
    def run():
        halo.img_u16_to_f32_axpby_2d(buf, width, height, stride, out, ostride, bit_depth)
    return _timeit(run)


def bench_box_blur(halo: HALO, width: int, height: int, radius: int) -> float:
    src, sstride = make_aligned_f32_buffer(width, height)
    dst, dstride = make_aligned_f32_buffer(width, height)
    def run():
        halo.box_blur_f32(src, dst, width, height, sstride, dstride, radius)
    return _timeit(run)


def bench_resize(halo: HALO, src_w: int, src_h: int, dst_w: int, dst_h: int) -> float:
    src, sstride = make_aligned_f32_buffer(src_w, src_h)
    dst, dstride = make_aligned_f32_buffer(dst_w, dst_h)
    def run():
        halo.resize_bilinear_f32(src, src_w, src_h, sstride, dst, dst_w, dst_h, dstride)
    return _timeit(run)


def collect_results(halo: HALO, sizes: Iterable[tuple[int, int]], threads: Iterable[int]) -> List[BenchResult]:
    results: List[BenchResult] = []
    for thr in threads:
        halo.set_threads(thr)
        for width, height in sizes:
            n = width * height
            results.append(BenchResult("saxpy", n, 1, thr, bench_saxpy(halo, n)))
            results.append(BenchResult("sum", n, 1, thr, bench_sum(halo, n)))
            results.append(BenchResult("img_u8", width, height, thr, bench_img_u8(halo, width, height)))
            results.append(BenchResult("img_u16", width, height, thr, bench_img_u16(halo, width, height)))
            results.append(BenchResult("box_blur_r3", width, height, thr, bench_box_blur(halo, width, height, 3)))
            dst_w = max(1, width // 2)
            dst_h = max(1, height // 2)
            results.append(BenchResult("resize_bilinear", dst_w, dst_h, thr, bench_resize(halo, width, height, dst_w, dst_h)))
    return results


def write_csv(results: Iterable[BenchResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "width", "height", "threads", "elapsed_s", "notes"])
        for r in results:
            writer.writerow([r.name, r.width, r.height, r.threads, f"{r.elapsed:.6f}", r.notes])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HALO Benchmark Suite")
    p.add_argument("--output", type=Path, default=Path("halo_bench.csv"), help="Zieldatei für CSV")
    p.add_argument("--threads", type=int, nargs="*", default=[1, 4], help="Thread-Anzahlen")
    p.add_argument("--sizes", type=str, nargs="*", default=["512x512", "1024x1024"], help="Breite×Höhe")
    p.add_argument("--force-autotune", action="store_true", help="Autotuning erzwingen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    size_list: List[tuple[int, int]] = []
    for s in args.sizes:
        if "x" not in s:
            raise ValueError(f"Ungültiges Format für size: {s}")
        w_str, h_str = s.lower().split("x", 1)
        size_list.append((int(w_str), int(h_str)))

    halo = HALO(force_autotune=args.force_autotune)
    results = collect_results(halo, size_list, args.threads)
    write_csv(results, args.output)
    print(f"Ergebnisse nach {args.output} geschrieben ({len(results)} Zeilen).")
if __name__ == "__main__":
    main()
