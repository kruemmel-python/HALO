#!/usr/bin/env python3
"""Einfache Bench-Suite für HALO: misst SAXPY & u8→f32-Konvertierung.

Erzeugt eine CSV-Ausgabe, die Größen, Thread-Konfigurationen und Laufzeiten enthält.
"""
from __future__ import annotations

import argparse
import csv
import time
from array import array
from typing import Iterable, List, Tuple

from halo import HALO, make_identity_lut


def parse_sizes(values: Iterable[str]) -> List[int]:
    sizes: List[int] = []
    for val in values:
        if "x" in val.lower():
            # Bildgröße -> Fläche
            w_str, h_str = val.lower().split("x", 1)
            w = int(w_str)
            h = int(h_str)
            sizes.append((w, h))
        else:
            sizes.append(int(val))
    return sizes


def bench_saxpy(halo: HALO, n: int, threads: int, iters: int) -> float:
    halo.set_threads(threads)
    x = array('f', [0.5] * n)
    y = array('f', [1.0] * n)
    halo.saxpy(0.1, x, y)  # Warmup
    start = time.perf_counter()
    for _ in range(iters):
        halo.saxpy_mt(0.1, x, y)
    end = time.perf_counter()
    return (end - start) / iters


def bench_img_u8(halo: HALO, size: Tuple[int, int], threads: int, iters: int) -> float:
    width, height = size
    halo.set_threads(threads)
    src = bytearray(width * height)
    dst = array('f', [0.0] * (width * height))
    lut = make_identity_lut()
    stride_src = width
    stride_dst = width * 4
    halo.img_u8_to_f32_lut_axpby_2d(src, width, height, stride_src, dst, stride_dst, lut,
                                    scale=1.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True)
    start = time.perf_counter()
    for _ in range(iters):
        halo.img_u8_to_f32_lut_axpby_2d(src, width, height, stride_src, dst, stride_dst, lut,
                                        scale=1.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True)
    end = time.perf_counter()
    return (end - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="HALO Benchmark Suite")
    parser.add_argument("--vector-sizes", nargs="*", default=["1000000", "5000000"],
                        help="Vektorgrößen für SAXPY (Default: 1e6, 5e6)")
    parser.add_argument("--image-sizes", nargs="*", default=["1920x1080", "3840x2160"],
                        help="Bildgrößen WxH für u8→f32 (Default: 1920x1080, 3840x2160)")
    parser.add_argument("--threads", nargs="*", default=["1", "2", "4"],
                        help="Thread-Anzahlen")
    parser.add_argument("--iters", type=int, default=5, help="Durchläufe pro Messung")
    parser.add_argument("--output", default="bench.csv", help="Zieldatei für CSV")
    args = parser.parse_args()

    vec_sizes = [int(s) for s in args.vector_sizes]
    img_sizes = [tuple(map(int, s.lower().split("x", 1))) for s in args.image_sizes]
    threads = [int(t) for t in args.threads]

    halo = HALO(force_autotune=False)

    rows = []
    for n in vec_sizes:
        for t in threads:
            dur = bench_saxpy(halo, n, t, args.iters)
            rows.append({
                "kernel": "saxpy_mt",
                "size": n,
                "width": "",
                "height": "",
                "threads": t,
                "time_s": f"{dur:.6f}",
            })
    for w, h in img_sizes:
        for t in threads:
            dur = bench_img_u8(halo, (w, h), t, args.iters)
            rows.append({
                "kernel": "img_u8_to_f32",
                "size": w * h,
                "width": w,
                "height": h,
                "threads": t,
                "time_s": f"{dur:.6f}",
            })

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kernel", "size", "width", "height", "threads", "time_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ergebnisse gespeichert in {args.output}")


+if __name__ == "__main__":
+    main()
