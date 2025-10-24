# all_ops_gallery.py
# Demonstriert (fast) alle Funktionen der HALO-DLL über halo.py (v0.5b):
#  - SAXPY/SUM
#  - u8/u16/RGB -> f32 Konvertierung (LUT/AXPBY, interleaved/planar)
#  - Box/Gaussian Blur, Sobel
#  - Resize (bilinear/bicubic)
#  - ReLU+Clamp AXPBY
#  - Flip/Rotate90, Invert, Gamma, Levels, Threshold
#  - Median/Erode/Dilate/Open/Close, Unsharp Mask
#  - Vektor-Rendering & Plot
#
# Erzeugt PNGs: 01_...png, 02_...png, ... im Arbeitsverzeichnis.

from __future__ import annotations
import time, math, os
import numpy as np
import imageio.v3 as iio

from halo import (
    HALO, Impl,
    make_aligned_u8_buffer,
    make_aligned_f32_buffer,
    make_pinned_float_array,
    make_identity_lut,
    VectorCanvas, VectorShape
)

# ---------------- Utilities ----------------

def save_f32(path: str, buf_mv, w: int, h: int, stride_bytes: int, *, lo=0.0, hi=1.0):
    """Speichert f32-Buffer (mit Stride) als 8-bit PNG."""
    arr = np.zeros((h, w), dtype=np.float32)
    row_pitch = stride_bytes // 4
    src = np.frombuffer(buf_mv, dtype=np.float32)
    for y in range(h):
        arr[y] = src[y*row_pitch : y*row_pitch + w]
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() if arr.max() > arr.min() else arr.min()+1.0)
    img = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    iio.imwrite(path, (img * 255.0 + 0.5).astype(np.uint8))

def save_rgb_f32(path: str, buf_mv, w: int, h: int, stride_bytes: int):
    """Speichert f32-RGB interleaved als PNG (0..1 erwartet)."""
    row_pitch = stride_bytes // 4
    src = np.frombuffer(buf_mv, dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        row = src[y*row_pitch : y*row_pitch + 3*w]
        img[y] = row.reshape(w,3)
    img8 = np.clip(img, 0.0, 1.0) * 255.0
    iio.imwrite(path, img8.astype(np.uint8))

def stamp(name: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"{name:<32s} {t1 - t0:6.3f}s")
    return t1

# ---------------- Start ----------------

if __name__ == "__main__":
    t0_all = time.perf_counter()

    # HALO initialisieren (0 => Auto Threadzahl)
    halo = HALO(threads=0, enable_streaming=True)
    print("HALO:", getattr(halo, "features", {}))

    # ------------- SAXPY / SUM -------------
    n = 1_000_000
    x = make_pinned_float_array(n)
    y = make_pinned_float_array(n)
    x_np = np.frombuffer(x, dtype=np.float32); x_np[:] = np.sin(np.linspace(0, 50, n)).astype(np.float32)
    y_np = np.frombuffer(y, dtype=np.float32); y_np[:] = 1.0

    t = time.perf_counter()
    halo.saxpy(0.25, x, y)        # ST
    s1 = halo.sum(y)              # SUM
    t = stamp("saxpy + sum (ST)", t)

    halo.set_impls(Impl.AVX2_STREAM, Impl.AVX2)  # nur um die Setter zu berühren
    halo.set_threads(0)           # ggf. Threads from profile
    t = time.perf_counter()
    halo.saxpy_mt(0.25, x, y)     # MT
    s2 = halo.sum(y)
    t = stamp("saxpy_mt + sum (MT)", t)
    print("SUMs:", s1, s2)

    # Bildgröße Basis
    W, H = 480, 320

    # ------------- u8 -> f32 (LUT + AXPBY) -------------
    u8, s_u8 = make_aligned_u8_buffer(W, H, alignment=64)  # 1 byte pro Pixel
    lut = make_identity_lut()
    # Testmuster: Schachbrett + Verlauf
    u8_np = np.frombuffer(u8, dtype=np.uint8)
    for y in range(H):
        for x0 in range(W):
            v = ((x0//16 + y//16) & 1) * 32 + (x0 * 223 // (W-1))
            u8_np[y*s_u8 + x0] = np.uint8(v)
    f32_lut, s_f32_lut = make_aligned_f32_buffer(W, H, alignment=64)
    t = time.perf_counter()
    halo.img_u8_to_f32_lut_axpby_2d(
        u8, W, H, s_u8,
        f32_lut, s_f32_lut,
        lut, scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True
    )
    t = stamp("img_u8_to_f32_lut_axpby", t)
    save_f32("01_u8_lut.png", f32_lut, W, H, s_f32_lut)

    # ------------- u16 -> f32 (AXPBY) -------------
    u16 = (np.linspace(0, 65535, W*H, dtype=np.uint16).reshape(H,W) ^ 0xAAAA).astype(np.uint16)
    u16b = memoryview(u16.tobytes())
    f32_u16, s_f32_u16 = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    halo.img_u16_to_f32_axpby_2d(
        u16b, W, H, W*2,
        f32_u16, s_f32_u16,
        bit_depth=16, scale=1.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True
    )
    t = stamp("img_u16_to_f32_axpby", t)
    save_f32("02_u16_to_f32.png", f32_u16, W, H, s_f32_u16, lo=0, hi=65535)

    # ------------- RGB (interleaved) -> f32 -------------
    # Interleaved RGB: einfache Farbbalken
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    rgb[...,0] = np.tile(np.linspace(0,255,W, dtype=np.uint8), (H,1))
    rgb[...,1] = np.roll(rgb[...,0], W//3, axis=1)
    rgb[...,2] = np.roll(rgb[...,0], 2*W//3, axis=1)
    rgb_b = memoryview(rgb.tobytes())
    f32_rgb, s_f32_rgb = make_aligned_f32_buffer(W, H, components=3)
    t = time.perf_counter()
    halo.img_rgb_u8_to_f32_interleaved(
        rgb_b, W, H, W*3,           # <— Reihenfolge & Stride in BYTES
        f32_rgb, s_f32_rgb,
        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True
    )
    t = stamp("img_rgb_u8_to_f32_interleaved", t)
    save_rgb_f32("03_rgb_interleaved.png", f32_rgb, W, H, s_f32_rgb)

    # ------------- RGB (planar) -> f32 -------------
    R = rgb[...,0].copy(order="C")
    G = rgb[...,1].copy(order="C")
    B = rgb[...,2].copy(order="C")
    r_b = memoryview(R)
    g_b = memoryview(G)
    b_b = memoryview(B)
    r_f32, s_r = make_aligned_f32_buffer(W, H)
    g_f32, s_g = make_aligned_f32_buffer(W, H)
    b_f32, s_b = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    halo.img_rgb_u8_to_f32_planar(
        r_b, g_b, b_b, W, H, W, W, W,
        r_f32, g_f32, b_f32, s_r, s_g, s_b,
        scale=1/255.0, offset=0.0, alpha=0.0, beta=1.0, use_mt=True
    )
    t = stamp("img_rgb_u8_to_f32_planar", t)
    # pack & save (nur zum Anschauen)
    packed, spack = make_aligned_f32_buffer(W, H, components=3)
    tmp = np.zeros((H,W,3), dtype=np.float32)
    tmp[...,0] = np.frombuffer(r_f32, np.float32).reshape(H,W)
    tmp[...,1] = np.frombuffer(g_f32, np.float32).reshape(H,W)
    tmp[...,2] = np.frombuffer(b_f32, np.float32).reshape(H,W)
    np.frombuffer(packed, np.float32)[:H*W*3] = tmp.ravel()
    save_rgb_f32("04_rgb_planar.png", packed, W, H, spack)

    # ------------- Filters & Geometrie -------------
    # Quelle: f32_lut (Graustufen 0..1)
    src = f32_lut; s_src = s_f32_lut
    dst, s_dst = make_aligned_f32_buffer(W, H)

    t = time.perf_counter()
    halo.box_blur_f32(src, dst, W, H, s_src, s_dst, radius=5, use_mt=True)
    t = stamp("box_blur_f32 (r=5)", t)
    save_f32("05_box_blur.png", dst, W, H, s_dst)

    src, dst = dst, src; s_src, s_dst = s_dst, s_src
    t = time.perf_counter()
    halo.gaussian_blur_f32(src, dst, W, H, s_src, s_dst, sigma=2.0, use_mt=True)
    t = stamp("gaussian_blur_f32 (σ=2.0)", t)
    save_f32("06_gaussian.png", dst, W, H, s_dst)

    src, dst = dst, src; s_src, s_dst = s_dst, s_src
    t = time.perf_counter()
    halo.sobel_f32(src, dst, W, H, s_src, s_dst, use_mt=True)
    t = stamp("sobel_f32", t)
    save_f32("07_sobel.png", dst, W, H, s_dst)

    # Resize (down & up)
    small_w, small_h = 240, 160
    f_small, s_small = make_aligned_f32_buffer(small_w, small_h)
    t = time.perf_counter()
    halo.resize_bilinear_f32(dst, W, H, s_dst, f_small, small_w, small_h, s_small, use_mt=True)
    t = stamp("resize_bilinear_f32 down", t)
    save_f32("08_resize_bilinear_down.png", f_small, small_w, small_h, s_small)

    big_w, big_h = 720, 480
    f_big, s_big = make_aligned_f32_buffer(big_w, big_h)
    t = time.perf_counter()
    halo.resize_bicubic_f32(dst, W, H, s_dst, f_big, big_w, big_h, s_big, use_mt=True)
    t = stamp("resize_bicubic_f32 up", t)
    save_f32("09_resize_bicubic_up.png", f_big, big_w, big_h, s_big)

    # ReLU + Clamp + AXPBY (mischt Original + Blur mit Grenzen)
    srcA = f32_lut; sA = s_f32_lut
    dstA, sDA = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    halo.relu_clamp_axpby_f32(srcA, dstA, W, H, sA, sDA,
                              alpha=0.8, beta=0.6, clamp_min=0.1, clamp_max=0.9,
                              apply_relu=True, use_mt=True)
    t = stamp("relu_clamp_axpby_f32", t)
    save_f32("10_relu_clamp.png", dstA, W, H, sDA)

    # ----- Neue Bild-OPs -----
    def maybe(op_name, fn):
        try:
            fn()
        except AttributeError:
            print(f"[skip] {op_name} (Wrapper-Methode fehlt)")
        except Exception as e:
            print(f"[err ] {op_name}: {e}")

    # invert_f32
    inv, s_inv = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("invert_f32", lambda: halo.invert_f32(srcA, inv, W, H, sA, s_inv, min_val=0.0, max_val=1.0, use_range=True, use_mt=True))
    t = stamp("invert_f32", t); save_f32("11_invert.png", inv, W, H, s_inv)

    # gamma_f32
    gam, s_gam = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("gamma_f32", lambda: halo.gamma_f32(srcA, gam, W, H, sA, s_gam, gamma=2.2, gain=1.0, use_mt=True))
    t = stamp("gamma_f32", t); save_f32("12_gamma.png", gam, W, H, s_gam)

    # levels_f32
    lev, s_lev = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("levels_f32", lambda: halo.levels_f32(srcA, lev, W, H, sA, s_lev, in_low=0.2, in_high=0.8, out_low=0.0, out_high=1.0, gamma=1.2, use_mt=True))
    t = stamp("levels_f32", t); save_f32("13_levels.png", lev, W, H, s_lev)

    # threshold_f32
    thr, s_thr = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("threshold_f32", lambda: halo.threshold_f32(srcA, thr, W, H, sA, s_thr, low=0.4, high=0.6, low_value=0.0, high_value=1.0, use_mt=True))
    t = stamp("threshold_f32", t); save_f32("14_threshold.png", thr, W, H, s_thr)

    # median3x3_f32
    med, s_med = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("median3x3_f32", lambda: halo.median3x3_f32(srcA, med, W, H, sA, s_med, use_mt=True))
    t = stamp("median3x3_f32", t); save_f32("15_median.png", med, W, H, s_med)

    # erode/dilate/open/close (3x3)
    ero, s_ero = make_aligned_f32_buffer(W, H)
    dil, s_dil = make_aligned_f32_buffer(W, H)
    opn, s_opn = make_aligned_f32_buffer(W, H)
    clo, s_clo = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("erode3x3_f32",  lambda: halo.erode3x3_f32(srcA, ero, W, H, sA, s_ero, use_mt=True))
    t = stamp("erode3x3_f32", t); save_f32("16_erode.png", ero, W, H, s_ero)
    t = time.perf_counter()
    maybe("dilate3x3_f32", lambda: halo.dilate3x3_f32(srcA, dil, W, H, sA, s_dil, use_mt=True))
    t = stamp("dilate3x3_f32", t); save_f32("17_dilate.png", dil, W, H, s_dil)
    t = time.perf_counter()
    maybe("open3x3_f32",   lambda: halo.open3x3_f32(srcA, opn, W, H, sA, s_opn, use_mt=True))
    t = stamp("open3x3_f32", t); save_f32("18_open.png", opn, W, H, s_opn)
    t = time.perf_counter()
    maybe("close3x3_f32",  lambda: halo.close3x3_f32(srcA, clo, W, H, sA, s_clo, use_mt=True))
    t = stamp("close3x3_f32", t); save_f32("19_close.png", clo, W, H, s_clo)

    # unsharp_mask_f32
    usm, s_usm = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("unsharp_mask_f32", lambda: halo.unsharp_mask_f32(srcA, usm, W, H, sA, s_usm, sigma=1.0, amount=1.5, threshold=0.01, use_mt=True))
    t = stamp("unsharp_mask_f32", t); save_f32("20_unsharp.png", usm, W, H, s_usm)

    # flip / rotate90
    flp, s_flp = make_aligned_f32_buffer(W, H)
    t = time.perf_counter()
    maybe("flip_f32", lambda: halo.flip_f32(srcA, flp, W, H, sA, s_flp, horizontal=True,  vertical=False, use_mt=True))
    t = stamp("flip_f32 (H)", t); save_f32("21_flip_h.png", flp, W, H, s_flp)
    t = time.perf_counter()
    maybe("flip_f32", lambda: halo.flip_f32(srcA, flp, W, H, sA, s_flp, horizontal=False, vertical=True,  use_mt=True))
    t = stamp("flip_f32 (V)", t); save_f32("22_flip_v.png", flp, W, H, s_flp)

    rot_w, rot_h = H, W  # 90°
    rot, s_rot = make_aligned_f32_buffer(rot_w, rot_h)
    t = time.perf_counter()
    maybe("rotate90_f32", lambda: halo.rotate90_f32(srcA, rot, W, H, sA, s_rot, quarter_turns=1, use_mt=True))
    t = stamp("rotate90_f32 (90°)", t); save_f32("23_rotate90.png", rot, rot_w, rot_h, s_rot)

    # ------------- Vektor-Rendering & Plot -------------
    # Vector: Herz + Dreieck + Text
    canvas = VectorCanvas(W, H, background=0.0)
    heart = VectorShape.from_svg(
        "M 240 80 C 240 40, 200 40, 200 80 C 200 110, 240 120, 240 160 "
        "C 240 120, 280 110, 280 80 C 280 40, 240 40, 240 80 Z",
        tolerance=0.5, fill_color=0.8, stroke_color=1.0, stroke_width=1.0
    )
    tri = VectorShape([[(60,220),(120,280),(0,280)]],[True],fill_color=0.5,stroke_color=1.0,stroke_width=1.5)
    canvas.draw_shape(heart); canvas.draw_shape(tri)
    canvas.draw_text(20, H-8, "HALO v0.5b", size=16, color=1.0)
    vec, s_vec = make_aligned_f32_buffer(W, H)
    canvas.blit_to(vec, s_vec)
    save_f32("24_vector_scene.png", vec, W, H, s_vec)

    # Plot
    xs = np.linspace(0, 6.2831, 400).astype(np.float32)
    ys = (0.5 + 0.5*np.sin(xs*2.0)).astype(np.float32)
    plot, s_plot = make_aligned_f32_buffer(W, H)
    halo.render_line_plot(xs, ys, plot, W, H, s_plot,
                          background=0.0, color=1.0, tick_count=5,
                          x_label="x", y_label="sin(2x)", title="HALO Plot")
    save_f32("25_plot.png", plot, W, H, s_plot)

    # ------------- RGB-Komposition (Bonus) -------------
    # Nimmt Graubild und mappt als RGB (R=relu/clamp, G=original, B=rotated-resized)
    rgb_mix, s_rgb_mix = make_aligned_f32_buffer(W, H, components=3)
    r = np.frombuffer(dstA, np.float32).reshape(H, W)      # relu/clamp
    g = np.frombuffer(f32_lut, np.float32).reshape(H, W)   # original
    b = np.frombuffer(rot, np.float32).reshape(rot_h, rot_w)
    # bring b auf W×H zurück
    b_res, s_bres = make_aligned_f32_buffer(W, H)
    halo.resize_bilinear_f32(rot, rot_w, rot_h, s_rot, b_res, W, H, s_bres)
    b = np.frombuffer(b_res, np.float32).reshape(H, W)
    mix = np.dstack([r, g, b]).astype(np.float32)
    np.frombuffer(rgb_mix, np.float32)[:W*H*3] = mix.ravel()
    save_rgb_f32("26_rgb_mix.png", rgb_mix, W, H, s_rgb_mix)

    # ------------- Ende -------------
    t_all = time.perf_counter() - t0_all
    print(f"\nFertig. Gesamt: {t_all:.3f}s")
