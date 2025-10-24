# test_img2d.py — v0.5 Benchmark & Check (1080p, affine LUT)
from __future__ import annotations
from time import perf_counter
from array import array
import os
from halo import HALO, make_identity_lut

USE_STREAMING = True
THREADS       = os.cpu_count() or 4
W, H          = 1920, 1080
SRC_STRIDE    = 2048             # >= W
DST_STRIDE    = ((W*4 + 63)//64)*64   # 64B-ausgerichtete Zeilenlänge hilft NT-Stores

def ref_kernel(src_bytes: bytes, W: int, H: int, sstride: int,
               dst: array, dstride: int, lut: list[float],
               scale: float, offset: float, alpha: float, beta: float):
    for y in range(H):
        srow = y * sstride
        drow = y * (dstride // 4)
        for x in range(W):
            v = src_bytes[srow + x]
            tmp = lut[v] * scale + offset
            dst[drow + x] = alpha * dst[drow + x] + beta * tmp

def bench(fn, repeats=7):
    t0 = perf_counter()
    for _ in range(repeats): fn()
    return (perf_counter() - t0) / repeats

if __name__ == "__main__":
    rng = os.urandom

    src = bytearray(rng(H * SRC_STRIDE))
    dst = array('f', [0.0]*(H * (DST_STRIDE // 4)))
    lut = make_identity_lut()

    halo = HALO(n_autotune=1_000_000, iters=5,
                enable_streaming=USE_STREAMING,
                stream_threshold=500_000, threads=THREADS)

    # Korrektheit (klein)
    w2, h2, s2, d2 = 64, 37, 80, 64*4
    src_small = bytearray(rng(h2 * s2))
    dst_small = array('f', [0.0]*(h2 * (d2 // 4)))
    halo.img_u8_to_f32_lut_axpby_2d(src_small, w2, h2, s2, dst_small, d2, lut,
                                    scale=1/255.0, offset=0.0, alpha=0.1, beta=0.9, use_mt=False)
    ref = array('f', [0.0]*(h2 * (d2 // 4)))
    ref_kernel(src_small, w2, h2, s2, ref, d2, list(lut), 1/255.0, 0.0, 0.1, 0.9)
    max_err = max(abs(a-b) for a,b in zip(dst_small, ref))
    print(f"[CHECK] max_abs_error = {max_err:.6g}")

    # Bench (v0.5 Auto-Scheduler: wir rufen ST/MT separat, Option use_mt übergeben)
    def run_st():
        halo.img_u8_to_f32_lut_axpby_2d(src, W, H, SRC_STRIDE, dst, DST_STRIDE, lut,
                                        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=False)
    def run_mt():
        halo.img_u8_to_f32_lut_axpby_2d(src, W, H, SRC_STRIDE, dst, DST_STRIDE, lut,
                                        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True)

    # Warmup
    run_st(); run_mt()

    dt_st = bench(run_st, repeats=9)
    dt_mt = bench(run_mt, repeats=9)

    gbps = lambda dt: (5 * W * H) / dt / 1e9
    mode = "ON" if USE_STREAMING else "OFF"
    print(f"[CFG ] streaming={mode}, threads={THREADS}")
    print(f"[ST ] {W}x{H} | {dt_st*1e3:7.3f} ms | ~{gbps(dt_st):6.2f} GB/s")
    print(f"[MT ] {W}x{H} | {dt_mt*1e3:7.3f} ms | ~{gbps(dt_mt):6.2f} GB/s")
