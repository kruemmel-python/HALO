import gradio as gr
import numpy as np
import math
import ctypes as C
import cv2 
from typing import Optional, Tuple, List, Union, Any, Literal
from array import array
import os
from pathlib import Path
import platform
import json

# ====================================================================
# ABSCHNITT 1: HALO-Klassen, Konvertierungs-Hilfsmittel und Imports
# ====================================================================

# Typdefinitionen
ArrayLikeFloat = Union[array, memoryview]
ArrayLikeU8    = Union[bytes, bytearray, memoryview]
HALO_Buffer = ArrayLikeFloat 

try:
    from halo import HALO, make_aligned_f32_buffer
    from halo_extensions import (
        bilateral_filter, canny_edge_detector, convert_colorspace, 
        morphological_operation as hl_morph_op, warp_affine, AffineTransform
    )
    
    def _to_c_f32_ptr_2d(buf: HALO_Buffer, *args, **kwargs) -> HALO_Buffer:
        return buf

    halo_instance = HALO(threads=4, use_gpu=True)
    print(f"HALO-Bibliothek ({halo_instance.features}) erfolgreich geladen.")
    if getattr(halo_instance, "gpu_enabled", False):
        print(f"GPU-Beschleunigung aktiviert (Gerät {halo_instance.gpu_device}).")
    else:
        print("GPU-Beschleunigung nicht verfügbar – CPU-Pfade aktiv.")
    
except ImportError as e:
    print(f"FEHLER: HALO-Import fehlgeschlagen. Einige Funktionen sind deaktiviert. {e}")
    class DummyHALO:
        def __init__(self): pass
        def __getattr__(self, name): return lambda *args, **kwargs: 0
    class DummyExtensions:
        def __getattr__(self, name): 
            return lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8)
    halo_instance = DummyHALO()
    hl_morph_op = bilateral_filter = canny_edge_detector = convert_colorspace = warp_affine = DummyExtensions().dummy
    AffineTransform = object()
    make_aligned_f32_buffer = lambda *args, **kwargs: (memoryview(bytearray(1)), 4)


# --- HALO-Bridge Logik ---

def _get_dtype_info(img: np.ndarray) -> Tuple[np.dtype, float, float]:
    if np.issubdtype(img.dtype, np.floating):
        return img.dtype, 1.0, 0.0
    if img.dtype == np.uint8:
        scale = 255.0
    elif img.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"Unsupported dtype: {img.dtype}")
    return img.dtype, scale, 0.0

def _normalize_to_float32(img: np.ndarray, dtype_info: Tuple[np.dtype, float, float]) -> np.ndarray:
    dtype, scale, _ = dtype_info
    if np.issubdtype(dtype, np.floating):
        return img.astype(np.float32)
    return img.astype(np.float32) / scale

def _denormalize_from_float32(img_f32: np.ndarray, dtype_info: Tuple[np.dtype, float, float]) -> np.ndarray:
    dtype, scale, _ = dtype_info
    if np.issubdtype(dtype, np.floating):
        return img_f32.astype(dtype, copy=False)
    
    max_val = scale
    img_scaled = np.clip(img_f32 * scale, 0, max_val)
    return img_scaled.astype(dtype, copy=False)

def _convert_image_to_float_channels(img: np.ndarray) -> List[Tuple[HALO_Buffer, int, Tuple[int, int]]]:
    if img is None: return []
    dtype_info = _get_dtype_info(img)
    img_f32 = _normalize_to_float32(img, dtype_info)
    h, w = img_f32.shape[:2]
    c = img_f32.shape[2] if img_f32.ndim == 3 else 1
    if img_f32.ndim == 2: img_f32 = img_f32[..., None]
    channels = np.split(img_f32, c, axis=2)
    halo_buffers = []
    for i in range(c):
        buf_mv, stride_bytes = make_aligned_f32_buffer(w, h, components=1, alignment=64)
        flat_data = channels[i].flatten()
        buf_mv[:w*h] = flat_data
        halo_buffers.append((buf_mv, stride_bytes, (w, h)))
    return halo_buffers

def _convert_float_channels_to_image(channels_info: List[Tuple[HALO_Buffer, int, Tuple[int, int]]], original_dtype_info: Tuple[np.dtype, float, float]) -> np.ndarray:
    if not channels_info: return np.zeros((100, 100, 3), dtype=np.uint8)
    w_out, h_out = channels_info[0][2]
    output_channels = []
    for buf, stride_bytes, _ in channels_info:
        flat_data = np.array(buf[:w_out*h_out], dtype=np.float32)
        channel_f32 = np.reshape(flat_data, (h_out, w_out, 1))
        output_channels.append(channel_f32)
    img_f32 = np.concatenate(output_channels, axis=2)
    img_final = _denormalize_from_float32(img_f32.squeeze(), original_dtype_info)
    if img_final.ndim == 2:
        if not np.issubdtype(original_dtype_info[0], np.floating):
            return np.stack([img_final] * 3, axis=-1)
    return img_final

def apply_halo_filter_per_channel(img: np.ndarray, halo_fn_wrapper: callable, resize_target: Optional[Tuple[int, int]] = None,) -> np.ndarray:
    if img is None: return None
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    dtype_info = _get_dtype_info(img)
    input_buffers_info = _convert_image_to_float_channels(img)
    if not input_buffers_info: return img
    out_w, out_h = resize_target if resize_target else (w, h)
    output_buffers_info = []
    for i in range(c):
        out_buf_mv, out_stride_bytes = make_aligned_f32_buffer(out_w, out_h, components=1, alignment=64)
        output_buffers_info.append((out_buf_mv, out_stride_bytes, (out_w, out_h)))
    for i in range(c):
        src_buf, src_stride, _ = input_buffers_info[i]
        dst_buf, dst_stride, _ = output_buffers_info[i]
        try:
            halo_fn_wrapper(
                src_buf, src_stride, src_w=w, src_h=h,
                dst_buf=dst_buf, dst_stride=dst_stride, dst_w=out_w, dst_h=out_h,
            )
        except Exception as e:
             print(f"HALO Laufzeitfehler im Kanal {i}: {e}")
             return img 
    return _convert_float_channels_to_image(output_buffers_info, dtype_info)


# ---------------------------------------------------------------------------
# ABSCHNITT 3: Gradio Wrapper für Low-Level C++-Funktionen (Definiert)
# ---------------------------------------------------------------------------
# Alle C-Kernel Wrapper müssen hier definiert werden, bevor sie in den .click-Handlern verwendet werden.
def process_blur_box(img: np.ndarray, radius: int) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.box_blur_f32(src, dst_buf, src_w, src_h, ss, dst_stride, radius, use_mt=True)
    return apply_halo_filter_per_channel(img, wrapper)

def process_blur_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.gaussian_blur_f32(src, dst_buf, src_w, src_h, ss, dst_stride, sigma, use_mt=True)
    return apply_halo_filter_per_channel(img, wrapper)

def process_filter_sobel(img: np.ndarray) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.sobel_f32(src, dst_buf, src_w, src_h, ss, dst_stride, use_mt=True)
    return apply_halo_filter_per_channel(img, wrapper)

def process_filter_median(img: np.ndarray) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.median3x3_f32(src, dst_buf, src_w, src_h, ss, dst_stride, use_mt=True)
    return apply_halo_filter_per_channel(img, wrapper)

def process_filter_unsharp(img: np.ndarray, sigma: float, amount: float, threshold: float) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.unsharp_mask_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            sigma=sigma, amount=amount, threshold=threshold, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_tonewheel_invert(img: np.ndarray, use_range: bool) -> np.ndarray:
    min_val = 0.0
    max_val = 1.0
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.invert_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            min_val=min_val, max_val=max_val, use_range=use_range, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_tonewheel_gamma(img: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.gamma_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            gamma=gamma, gain=gain, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_tonewheel_levels(img: np.ndarray, in_low: float, in_high: float, gamma: float) -> np.ndarray:
    out_low = 0.0
    out_high = 1.0
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.levels_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            in_low=in_low, in_high=in_high, 
            out_low=out_low, out_high=out_high, 
            gamma=gamma, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_tonewheel_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    low_value = 0.0
    high_value = 1.0
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.threshold_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            low=low, high=high, low_value=low_value, high_value=high_value, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_morphology(img: np.ndarray, operation: str) -> np.ndarray:
    if operation == "Erode": halo_fn = halo_instance.erode3x3_f32
    elif operation == "Dilate": halo_fn = halo_instance.dilate3x3_f32
    elif operation == "Open": halo_fn = halo_instance.open3x3_f32
    elif operation == "Close": halo_fn = halo_instance.close3x3_f32
    else: return img
    if halo_fn is None: return img
    def morph_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_fn(src, dst_buf, src_w, src_h, ss, dst_stride, use_mt=True)
    return apply_halo_filter_per_channel(img, morph_wrapper)

def process_geometry_flip(img: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.flip_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            horizontal=horizontal, vertical=vertical, use_mt=True
        )
    return apply_halo_filter_per_channel(img, wrapper)

def process_geometry_rotate(img: np.ndarray, turns: int) -> np.ndarray:
    if halo_instance.rotate90_f32 is None: return img
    turns = int(turns) % 4
    if turns == 0: return img
    h, w = img.shape[:2]
    target_w, target_h = (h, w) if turns % 2 != 0 else (w, h)
    def rotate_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, dst_w, dst_h, **kwargs):
        halo_instance.rotate90_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            quarter_turns=turns, use_mt=True
        )
    return apply_halo_filter_per_channel(img, rotate_wrapper, resize_target=(target_w, target_h))

def process_geometry_resize(img: np.ndarray, target_w: int, target_h: int, interpolation: str) -> np.ndarray:
    if img is None: return None
    halo_fn = halo_instance.resize_bicubic_f32 if interpolation == "Bicubic" else halo_instance.resize_bilinear_f32
    if halo_fn is None: return img
    def resize_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, dst_w, dst_h, **kwargs):
        halo_fn(src, src_w, src_h, ss, dst_buf, dst_w, dst_h, dst_stride, use_mt=True)
    return apply_halo_filter_per_channel(img, resize_wrapper, resize_target=(target_w, target_h))

def process_custom_axpby(img: np.ndarray, alpha: float, beta: float, clamp_min: float, clamp_max: float, apply_relu: bool) -> np.ndarray:
    if halo_instance.relu_clamp_axpby_f32 is None: return img
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    dtype_info = _get_dtype_info(img)
    input_buffers_info = _convert_image_to_float_channels(img)
    output_buffers_info = []
    for i in range(c):
        out_buf_mv, out_stride_bytes = make_aligned_f32_buffer(w, h, components=1, alignment=64)
        out_buf_mv[:w*h] = input_buffers_info[i][0][:w*h] 
        output_buffers_info.append((out_buf_mv, out_stride_bytes, (w, h)))
    def axpby_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.relu_clamp_axpby_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            alpha=alpha, beta=beta, clamp_min=clamp_min, clamp_max=clamp_max, 
            apply_relu=apply_relu, use_mt=True
        )
    for i in range(c):
        src_buf, src_stride, _ = input_buffers_info[i]
        dst_buf, dst_stride, _ = output_buffers_info[i]
        try:
            axpby_wrapper(src_buf, src_stride, w, h, dst_buf, dst_stride)
        except Exception as e:
            print(f"HALO Laufzeitfehler in relu_clamp_axpby_f32: {e}")
            return img
    return _convert_float_channels_to_image(output_buffers_info, dtype_info)


# ---------------------------------------------------------------------------
# ABSCHNITT 4: Gradio Wrapper für High-Level Extensions
# ---------------------------------------------------------------------------

def apply_high_level_filter(img: np.ndarray, hl_fn: callable, *args, **kwargs) -> np.ndarray:
    if img is None: return None
    dtype_info = _get_dtype_info(img)
    img_hl_input = _normalize_to_float32(img, dtype_info) 
    
    try:
        result_f32 = hl_fn(img_hl_input, *args, **kwargs)
        return _denormalize_from_float32(result_f32, dtype_info)
    except Exception as e:
        print(f"High-Level-Fehler in {hl_fn.__name__}: {e}")
        return img 

def process_hl_bilateral(img: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return apply_high_level_filter(img, bilateral_filter, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)

def process_hl_canny(img: np.ndarray, low_threshold: float, high_threshold: float, sigma: float) -> np.ndarray:
    return apply_high_level_filter(img, canny_edge_detector, low_threshold=low_threshold, high_threshold=high_threshold, gaussian_sigma=sigma)

def process_hl_morphology(img: np.ndarray, operation: str) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return apply_high_level_filter(img, hl_morph_op, operation=operation, kernel=kernel)

def process_hl_colorspace(img: np.ndarray, src: str, dst: str) -> np.ndarray:
    return apply_high_level_filter(img, convert_colorspace, src=src, dst=dst)

def process_hl_warp_affine(img: np.ndarray, tx: float, ty: float, rot: float, scale: float) -> np.ndarray:
    if AffineTransform is object: return img
    transform = AffineTransform.from_components(translation=(tx, ty), rotation_deg=rot, scale=(scale, scale))
    return apply_high_level_filter(img, warp_affine, matrix=transform.matrix)


# ---------------------------------------------------------------------------
# ABSCHNITT 5: Echtzeit-Videoverarbeitung
# ---------------------------------------------------------------------------

# Korrigierte Version von process_video_canny_pipeline

def process_video_canny_pipeline(video_path: str, low_th: float, high_th: float, sigma: float) -> str:
    """Verarbeitet ein Video Frame für Frame mit dem HALO Canny Edge Detector."""
    if cv2 is None: raise RuntimeError("OpenCV (cv2) ist für die Videoverarbeitung erforderlich. Bitte installieren Sie es.")
    if video_path is None: return None
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video-Datei nicht gefunden: {video_path}")
        
    # NEU: Ändere die Endung zu .avi für den universelleren MJPG Codec
    output_path = str(Path(video_path).parent / f"processed_canny_{Path(video_path).stem}.avi")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Video-Datei konnte nicht geöffnet werden.")

    # Video-Eigenschaften auslesen
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Reduziere die Auflösung für die Verarbeitung auf 50% (lastet 1/4 der Pixel)
    target_width = width // 2
    target_height = height // 2
    
    # NEU: Codec auf MJPG umstellen (universeller, robuster als MP4V auf vielen Windows-Setups)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height), isColor=True)
    
    # WICHTIG: Prüfe, ob der Writer initialisiert wurde
    if not out.isOpened():
         raise IOError(f"VideoWriter konnte nicht initialisiert werden. Codec MJPG oder Pfad {output_path} prüfen.")


    print(f"Starte Videoverarbeitung: {width}x{height} -> {target_width}x{target_height} @ {fps:.1f} FPS")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 1. HALO Canny Edge Detection (auf dem kleineren Frame)
        dtype_info = _get_dtype_info(frame_rgb)
        frame_f32 = _normalize_to_float32(frame_rgb, dtype_info)
        
        edges_f32 = canny_edge_detector(
            frame_f32,
            low_threshold=low_th,
            high_threshold=high_th,
            gaussian_sigma=sigma
        )
        
        # 2. Denormalisierung und Finalisierung
        edges_final = _denormalize_from_float32(edges_f32, dtype_info)
        
        # 3. Für den Video-Writer: Graustufen auf 3 Kanäle erweitern (muss BGR sein)
        if edges_final.ndim == 2:
            edges_final_rgb = np.stack([edges_final] * 3, axis=-1)
        else:
             edges_final_rgb = edges_final
            
        frame_bgr = cv2.cvtColor(edges_final_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        frame_count += 1
        if frame_count % 100 == 0:
             print(f"  > Verarbeitet {frame_count} Frames...")

    cap.release()
    out.release()
    print(f"Videoverarbeitung abgeschlossen. Total Frames: {frame_count}")
    
    return output_path


# ====================================================================
# ABSCHNITT 6: Gradio UI Definition
# ====================================================================

with gr.Blocks(title="HALO Image Processor Demo") as demo:
    gr.Markdown(
        """
        # HALO Image Processor Demo (CPU-Optimiert + High-Level Extensions)
        """
    )
    
    # ----------------------------------------------------------------
    # UI-Komponenten (Definitionen)
    # ----------------------------------------------------------------
    with gr.Row():
        image_input = gr.Image(label="Eingabebild (uint8/uint16/float)", type="numpy", interactive=True)
        image_output = gr.Image(label="Ergebnisbild", type="numpy")

    with gr.Tabs() as tabs:
        
        with gr.TabItem("C++ Filter (AVX2/MT)") as tab_c_filter:
            with gr.Column():
                gr.Markdown("### Faltungs- und Medianfilter")
                
                with gr.Row():
                    radius_box = gr.Slider(label="Box Blur Radius", minimum=1, maximum=20, step=1, value=3)
                    btn_box = gr.Button("Box Blur Anwenden")
                with gr.Row():
                    sigma_gauss = gr.Slider(label="Gaussian Blur Sigma", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                    btn_gauss = gr.Button("Gaussian Blur Anwenden")
                with gr.Row():
                    btn_sobel = gr.Button("Sobel Kantenfilter Anwenden")
                    btn_median = gr.Button("Median 3x3 Anwenden")
                with gr.Row():
                    gr.Markdown("#### Unsharp Masking")
                    sigma_unsharp = gr.Slider(label="Sigma", minimum=0.1, maximum=5.0, step=0.1, value=1.5)
                    amount_unsharp = gr.Slider(label="Amount", minimum=0.1, maximum=5.0, step=0.1, value=1.5)
                    threshold_unsharp = gr.Slider(label="Threshold", minimum=0.0, maximum=0.2, step=0.01, value=0.05)
                    btn_unsharp = gr.Button("Unsharp Mask Anwenden")

        with gr.TabItem("High-Level Filter (Python/NumPy)") as tab_hl_filter:
            gr.Markdown("### Fortgeschrittene Filter & Erweiterte Morphologie")
            with gr.Row():
                gr.Markdown("#### Bilateral Filter")
                bilat_diam = gr.Slider(label="Diameter", minimum=3, maximum=21, step=2, value=5)
                bilat_sigma_c = gr.Slider(label="Sigma Color", minimum=0.01, maximum=0.5, step=0.01, value=0.1)
                bilat_sigma_s = gr.Slider(label="Sigma Space", minimum=0.5, maximum=10.0, step=0.5, value=2.0)
                btn_bilat = gr.Button("Bilateral Filter Anwenden")
            with gr.Row():
                gr.Markdown("#### Canny Edge Detector")
                canny_low = gr.Slider(label="Low Threshold", minimum=0.01, maximum=0.5, step=0.01, value=0.1)
                canny_high = gr.Slider(label="High Threshold", minimum=0.1, maximum=0.9, step=0.01, value=0.3)
                canny_sigma = gr.Slider(label="Gaussian Sigma", minimum=0.5, maximum=3.0, step=0.1, value=1.4)
                btn_canny = gr.Button("Canny Edge Detector Anwenden")
            with gr.Row():
                gr.Markdown("#### Erweiterte Morphologie")
                hl_morph_op_type = gr.Radio(label="Operation", choices=["gradient", "tophat", "blackhat", "hitmiss"], value="gradient")
                btn_hl_morph = gr.Button("Morphologische Operation Anwenden (3x3)")

        with gr.TabItem("Farbraum & Warping") as tab_cs_warp:
            gr.Markdown("### Farbraum-Konvertierungen und Geometrie")
            with gr.Row():
                cs_src = gr.Radio(label="Quell-Farbraum", choices=["rgb", "hsv", "ycbcr"], value="rgb", type="value")
                cs_dst = gr.Radio(label="Ziel-Farbraum", choices=["rgb", "gray", "hsv", "ycbcr"], value="hsv", type="value")
                btn_cs = gr.Button("Farbraum Konvertieren")
            with gr.Row():
                gr.Markdown("#### Affine Warping")
                warp_tx = gr.Slider(label="Trans. X (Pixel)", minimum=-50, maximum=50, step=1, value=0)
                warp_ty = gr.Slider(label="Trans. Y (Pixel)", minimum=-50, maximum=50, step=1, value=0)
                warp_rot = gr.Slider(label="Rotation (°)", minimum=-180, maximum=180, step=1, value=0)
                warp_scale = gr.Slider(label="Skalierung", minimum=0.5, maximum=2.0, step=0.1, value=1.0)
                btn_warp = gr.Button("Affine Warp Anwenden")
                
        with gr.TabItem("Video-Streaming-Demo") as tab_video:
            gr.Markdown("### Echtzeit-Video-Pipeline (Canny Edge Detection)")
            gr.Markdown("Verarbeitet die Frames des hochgeladenen Videos mit dem Canny-Filter.")
            with gr.Row():
                video_input = gr.Video(label="Video-Datei hochladen", sources=["upload"])
                video_output = gr.Video(label="Verarbeitetes Video")
            with gr.Row():
                video_canny_low = gr.Slider(label="Low Threshold", minimum=0.01, maximum=0.5, step=0.01, value=0.1)
                video_canny_high = gr.Slider(label="High Threshold", minimum=0.1, maximum=0.9, step=0.01, value=0.3)
                video_canny_sigma = gr.Slider(label="Gaussian Sigma", minimum=0.5, maximum=3.0, step=0.1, value=1.4)
                btn_process_video = gr.Button("Video mit Canny Verarbeiten")
                
        with gr.TabItem("C++ Tonwert") as tab_c_tonewheel:
            gr.Markdown("### Helligkeits- und Farbraumkorrekturen (Float-Bereich 0-1)")
            with gr.Row():
                btn_invert = gr.Button("Invertieren")
            with gr.Row():
                gamma_gamma = gr.Slider(label="Gamma", minimum=0.1, maximum=5.0, step=0.1, value=2.2)
                gamma_gain = gr.Slider(label="Gain", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                btn_gamma = gr.Button("Gamma Anwenden")
            with gr.Row():
                levels_low = gr.Slider(label="Input Low (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.1)
                levels_high = gr.Slider(label="Input High (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                levels_gamma = gr.Slider(label="Gamma", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                btn_levels = gr.Button("Levels Anwenden")
            with gr.Row():
                threshold_low = gr.Slider(label="Threshold Low (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.4)
                threshold_high = gr.Slider(label="Threshold High (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.6)
                btn_threshold = gr.Button("Threshold Anwenden")
        
        with gr.TabItem("C++ Morphologie") as tab_c_morphology:
            gr.Markdown("### Morphologische 3x3 Operationen (Zwei-Pass Separabel)")
            with gr.Row():
                btn_erode = gr.Button("Erode 3x3 (Erosion)")
                btn_dilate = gr.Button("Dilate 3x3 (Dilatation)")
            with gr.Row():
                btn_open = gr.Button("Open 3x3 (Erode dann Dilate)")
                btn_close = gr.Button("Close 3x3 (Dilate dann Erode)")

        with gr.TabItem("C++ Geometrie") as tab_c_geometry:
            gr.Markdown("### Geometrische Operationen")
            with gr.Row():
                flip_h = gr.Checkbox(label="Horizontal spiegeln", value=True)
                flip_v = gr.Checkbox(label="Vertikal spiegeln", value=False)
                btn_flip = gr.Button("Spiegeln (Flip)")
            with gr.Row():
                rotate_turns = gr.Radio(label="90° Drehungen", choices=[1, 2, 3], value=1, type="value")
                btn_rotate = gr.Button("Drehen um 90°/180°/270°")
            with gr.Row():
                resize_w = gr.Slider(label="Neue Breite", minimum=64, maximum=1024, step=16, value=512)
                resize_h = gr.Slider(label="Neue Höhe", minimum=64, maximum=1024, step=16, value=512)
                resize_interp = gr.Radio(label="Interpolation", choices=["Bilinear", "Bicubic"], value="Bilinear")
                btn_resize = gr.Button("Bildgröße ändern")

        with gr.TabItem("C++ AXPBY/ReLU/Clamp") as tab_c_axpby:
            gr.Markdown("### Erweiterte AXPBY-Operation")
            with gr.Row():
                alpha_val = gr.Slider(label="Alpha (Gewichtung dst_orig)", minimum=0.0, maximum=2.0, step=0.01, value=0.0)
                beta_val = gr.Slider(label="Beta (Gewichtung src_orig)", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
            with gr.Row():
                clamp_min_val = gr.Slider(label="Clamp Min (0.0)", minimum=-0.5, maximum=0.5, step=0.01, value=0.0)
                clamp_max_val = gr.Slider(label="Clamp Max (1.0)", minimum=0.5, maximum=1.5, step=0.01, value=1.0)
                apply_relu_check = gr.Checkbox(label="Apply ReLU (max(0, val))", value=False)
                btn_axpby = gr.Button("AXPBY Anwenden")

    # ----------------------------------------------------------------
    # Klick-Handler (Registrierung der Ereignisse)
    # ----------------------------------------------------------------

    # C++ Filter
    btn_box.click(process_blur_box, inputs=[image_input, radius_box], outputs=image_output)
    btn_gauss.click(process_blur_gaussian, inputs=[image_input, sigma_gauss], outputs=image_output)
    btn_sobel.click(process_filter_sobel, inputs=[image_input], outputs=image_output)
    btn_median.click(process_filter_median, inputs=[image_input], outputs=image_output)
    btn_unsharp.click(process_filter_unsharp, inputs=[image_input, sigma_unsharp, amount_unsharp, threshold_unsharp], outputs=image_output)

    # High-Level Filter
    btn_bilat.click(process_hl_bilateral, inputs=[image_input, bilat_diam, bilat_sigma_c, bilat_sigma_s], outputs=image_output)
    btn_canny.click(process_hl_canny, inputs=[image_input, canny_low, canny_high, canny_sigma], outputs=image_output)
    btn_hl_morph.click(process_hl_morphology, inputs=[image_input, hl_morph_op_type], outputs=image_output)
    
    # Farbraum & Warping
    btn_cs.click(process_hl_colorspace, inputs=[image_input, cs_src, cs_dst], outputs=image_output)
    btn_warp.click(process_hl_warp_affine, inputs=[image_input, warp_tx, warp_ty, warp_rot, warp_scale], outputs=image_output)

    # Video-Verarbeitung (NEU)
    btn_process_video.click(
        process_video_canny_pipeline, 
        inputs=[video_input, video_canny_low, video_canny_high, video_canny_sigma],
        outputs=video_output
    )

    # C++ Tonwert
    btn_invert.click(process_tonewheel_invert, inputs=[image_input, gr.State(True)], outputs=image_output)
    btn_gamma.click(process_tonewheel_gamma, inputs=[image_input, gamma_gamma, gamma_gain], outputs=image_output)
    btn_levels.click(process_tonewheel_levels, inputs=[image_input, levels_low, levels_high, levels_gamma], outputs=image_output)
    btn_threshold.click(process_tonewheel_threshold, inputs=[image_input, threshold_low, threshold_high], outputs=image_output)

    # C++ Morphologie
    btn_erode.click(process_morphology, inputs=[image_input, gr.State("Erode")], outputs=image_output)
    btn_dilate.click(process_morphology, inputs=[image_input, gr.State("Dilate")], outputs=image_output)
    btn_open.click(process_morphology, inputs=[image_input, gr.State("Open")], outputs=image_output)
    btn_close.click(process_morphology, inputs=[image_input, gr.State("Close")], outputs=image_output)
    
    # C++ Geometrie
    btn_flip.click(process_geometry_flip, inputs=[image_input, flip_h, flip_v], outputs=image_output)
    btn_rotate.click(process_geometry_rotate, inputs=[image_input, rotate_turns], outputs=image_output)
    btn_resize.click(process_geometry_resize, inputs=[image_input, resize_w, resize_h, resize_interp], outputs=image_output)

    # C++ AXPBY
    btn_axpby.click(process_custom_axpby, 
                     inputs=[image_input, alpha_val, beta_val, clamp_min_val, clamp_max_val, apply_relu_check], 
                     outputs=image_output)


# Starte die Gradio App
if __name__ == "__main__":
    # Sicherstellen, dass cv2 importiert wurde (nur für den Video-Tab nötig)
    if 'cv2' not in locals():
        print("\nWARNUNG: OpenCV (cv2) konnte nicht importiert werden. Der Video-Tab wird nicht funktionieren.")
        print("         Bitte installieren Sie es mit 'pip install opencv-python'.")

    demo.launch()
