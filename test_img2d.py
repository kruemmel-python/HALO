# test_img2d.py — 2D-Image-Kern: Korrektheit & Benchmark
from __future__ import annotations
from array import array
from time import perf_counter
import os, random
from halo import HALO, make_identity_lut

def ref_kernel(src_bytes: bytes, W: int, H: int, sstride: int,
               dst: array, dstride: int, lut: list[float],
               scale: float, offset: float, alpha: float, beta: float):
    # Python-Referenz: einfach, korrekt, langsam
    for y in range(H):
        srow = y * sstride
        drow = y * (dstride // 4)
        for x in range(W):
            v = src_bytes[srow + x]
            tmp = lut[v] * scale + offset
            dst[drow + x] = alpha * dst[drow + x] + beta * tmp

def bench(fn, repeats=5):
    t0 = perf_counter()
    for _ in range(repeats):
        fn()
    return (perf_counter() - t0) / repeats

if __name__ == "__main__":
    W, H = 1920, 1080
    sstride = 2048       # bewusst größer als W (Padding)
    dstride = W * 4      # float32 pro Pixel
    # Erzeuge Testdaten
    src = bytearray(os.urandom(H * sstride))
    dst = array('f', [0.0]*(H * (dstride // 4)))

    # LUT: identity
    lut = make_identity_lut()

    # HALO initialisieren (mit MT)
    halo = HALO(n_autotune=1_000_000, iters=5, enable_streaming=True, stream_threshold=500_000, threads=os.cpu_count() or 4)

    # Korrektheit an kleinem Bild prüfen
    w2, h2 = 64, 37
    s2 = 80
    d2 = w2 * 4
    src_small = bytearray(os.urandom(h2 * s2))
    dst_small = array('f', [0.0]*(h2 * (d2 // 4)))
    # Run HALO
    halo.img_u8_to_f32_lut_axpby_2d(src_small, w2, h2, s2, dst_small, d2, lut,
                                    scale=1/255.0, offset=0.0, alpha=0.1, beta=0.9, use_mt=False)
    # Referenz
    ref = array('f', [0.0]*(h2 * (d2 // 4)))
    ref_kernel(src_small, w2, h2, s2, ref, d2, list(lut), 1/255.0, 0.0, 0.1, 0.9)
    # Check
    max_err = max(abs(a-b) for a,b in zip(dst_small, ref))
    print(f"[CHECK] max_abs_error = {max_err:.6g}")
    assert max_err < 1e-5

    # Benchmark groß: ST vs. MT
    def run_st():
        halo.img_u8_to_f32_lut_axpby_2d(src, W, H, sstride, dst, dstride, lut,
                                        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=False)

    def run_mt():
        halo.img_u8_to_f32_lut_axpby_2d(src, W, H, sstride, dst, dstride, lut,
                                        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True)

    # Warmups
    run_st(); run_mt()

    dt_st = bench(run_st, repeats=5)
    dt_mt = bench(run_mt, repeats=5)

    # Durchsatz grob: read src (1B) + write dst (4B) ~ 5B/Pixel + LUT/Gather/Arith.
    gbps_st = (5 * W * H) / dt_st / 1e9
    gbps_mt = (5 * W * H) / dt_mt / 1e9
    print(f"[ST ] {W}x{H} | {dt_st*1e3:7.3f} ms | ~{gbps_st:6.2f} GB/s")
    print(f"[MT ] {W}x{H} | {dt_mt*1e3:7.3f} ms | ~{gbps_mt:6.2f} GB/s")
