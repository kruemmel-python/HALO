import gradio as gr
import numpy as np
import math
import ctypes as C
from typing import Optional, Tuple, List, Union, Any
from array import array
import os
from pathlib import Path
import platform
import json

# ====================================================================
# ABSCHNITT 1: HALO-Klassen & ctypes-Adapter (Auszug aus halo.py)
# WICHTIG: Ersetzt durch dein vollständiges halo.py Modul.
# ====================================================================

# Typdefinitionen aus halo.py
ArrayLikeFloat = Union[array, memoryview]
ArrayLikeU8    = Union[bytes, bytearray, memoryview]
HALO_Buffer = ArrayLikeFloat # Für die interne Verarbeitung

# Hier wird der Code deines halo.py Moduls implizit angenommen. 
# WICHTIG: Stelle sicher, dass die echte halo.py und die DLL/SO im Pfad sind.

try:
    from halo import HALO, make_aligned_f32_buffer
    
    # Init-Logik zur Lasterkennung
    if 'HALO_Buffer' not in locals():
        HALO_Buffer = ArrayLikeFloat
    
    # Dummy-Funktion zur Umgehung der C-Pointer-Komplexität in den Wrappern
    def _to_c_f32_ptr_2d(buf: HALO_Buffer, *args, **kwargs) -> HALO_Buffer:
        # Im Gradio-Kontext (wenn NumPy-Array zu MemoryView konvertiert)
        # wird einfach das memoryview-Objekt selbst zurückgegeben.
        return buf

    halo_instance = HALO(threads=4)
    print(f"HALO-Bibliothek ({halo_instance.features}) erfolgreich geladen.")
    
except ImportError:
    print("FEHLER: Konnte HALO-Modul nicht importieren. Erzeuge Dummy-Klasse.")
    class DummyHALO:
        def __init__(self): pass
        def __getattr__(self, name):
            if name.startswith('resize'):
                return lambda src, w, h, ss, dst, dw, dh, ds, use_mt: 0
            if name.endswith('f32'):
                return lambda src, dst=None, w=0, h=0, ss=0, ds=0, *args, **kwargs: 0
            return lambda *args, **kwargs: None
    halo_instance = DummyHALO()
    
# --- Ende des HALO-Wrapper-Abschnitts ---


# ====================================================================
# ABSCHNITT 2: Gradio-Bridge und Konvertierungslogik (KORRIGIERT)
# ====================================================================

def _convert_image_to_float_channels(img: np.ndarray) -> List[Tuple[HALO_Buffer, int]]:
    """Konvertiert HxWxC uint8-Bild in eine Liste von (Channel-Buffer, Stride) float32-Arrays (0.0-1.0)."""
    if img is None: return []
    
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    h, w, c = img.shape
    img_f32 = img.astype(np.float32) / 255.0
    channels = np.split(img_f32, c, axis=2)
    
    halo_buffers = []
    for i in range(c):
        buf_mv, stride_bytes = make_aligned_f32_buffer(w, h, components=1, alignment=64)
        flat_data = channels[i].flatten()
        buf_mv[:w*h] = flat_data
        halo_buffers.append((buf_mv, stride_bytes))

    return halo_buffers

def _convert_float_channels_to_image(channels_info: List[Tuple[HALO_Buffer, int]], h: int, w: int) -> np.ndarray:
    """Konvertiert Liste von (Channel-Buffer, Stride) float32-Arrays zurück in HxWxC uint8-Bild."""
    if not channels_info:
        # Wenn wir die Höhe/Breite nicht kennen, Fallback auf Standard
        return np.zeros((512, 512, 3), dtype=np.uint8) if h == 0 or w == 0 else np.zeros((h, w, 3), dtype=np.uint8)

    output_channels = []
    # Die tatsächliche Höhe/Breite ergibt sich aus dem ersten Puffer und dessen Dimensionierung
    w_out = w
    h_out = h
    
    for buf, stride_bytes in channels_info:
        # Wir nehmen die Dimensionen des Buffers, da sie sich durch z.B. Rotation geändert haben können.
        flat_data = np.array(buf[:w_out*h_out], dtype=np.float32)
        channel_f32 = np.clip(np.reshape(flat_data, (h_out, w_out, 1)), 0.0, 1.0)
        output_channels.append(channel_f32)

    img_f32 = np.concatenate(output_channels, axis=2)
    img_uint8 = (img_f32 * 255.0).astype(np.uint8)
    return img_uint8


def apply_halo_filter_per_channel(
    img: np.ndarray,
    halo_fn_wrapper: callable,
    resize_target: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Basis-Wrapper, der Konvertierung, Pufferallokation und Kanalschleife handhabt."""
    if img is None: return None

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    
    input_buffers_info = _convert_image_to_float_channels(img)
    if not input_buffers_info: return img

    out_w, out_h = resize_target if resize_target else (w, h)

    output_buffers_info = []
    for i in range(c):
        out_buf_mv, out_stride_bytes = make_aligned_f32_buffer(out_w, out_h, components=1, alignment=64)
        output_buffers_info.append((out_buf_mv, out_stride_bytes))

    for i in range(c):
        src_buf, src_stride = input_buffers_info[i]
        dst_buf, dst_stride = output_buffers_info[i]

        try:
            halo_fn_wrapper(
                src_buf, src_stride, src_w=w, src_h=h,
                dst_buf=dst_buf, dst_stride=dst_stride, dst_w=out_w, dst_h=out_h,
            )
        except Exception as e:
             print(f"HALO Laufzeitfehler im Kanal {i}: {e}")
             return img 

    return _convert_float_channels_to_image(output_buffers_info, out_h, out_w)


# --- Spezifische HALO-Funktionen (mit Korrigierter Argumentübergabe) ---

# --- Blur/Filter ---

def process_blur_box(img: np.ndarray, radius: int) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.box_blur_f32(src, dst_buf, src_w, src_h, ss, dst_stride, radius, use_mt=True)
    return apply_halo_filter_per_channel(img, wrapper)

def process_blur_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    def wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        # Korrigiert: sigma ist Positional
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

# --- Tonwert ---

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

# --- Morphologie ---
def process_morphology(img: np.ndarray, operation: str) -> np.ndarray:
    if img is None: return None
    
    if operation == "Erode": halo_fn = halo_instance.erode3x3_f32
    elif operation == "Dilate": halo_fn = halo_instance.dilate3x3_f32
    elif operation == "Open": halo_fn = halo_instance.open3x3_f32
    elif operation == "Close": halo_fn = halo_instance.close3x3_f32
    else: return img
    
    if halo_fn is None: return img
    
    def morph_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_fn(src, dst_buf, src_w, src_h, ss, dst_stride, use_mt=True)
        
    return apply_halo_filter_per_channel(img, morph_wrapper)

# --- Geometrie ---

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
        # Resize-Signatur (src, src_w, src_h, ss, dst, dst_w, dst_h, ds, use_mt)
        halo_fn(src, src_w, src_h, ss, dst_buf, dst_w, dst_h, dst_stride, use_mt=True)

    return apply_halo_filter_per_channel(img, resize_wrapper, resize_target=(target_w, target_h))

# --- Custom: AXPBY ---

def process_custom_axpby(img: np.ndarray, alpha: float, beta: float, clamp_min: float, clamp_max: float, apply_relu: bool) -> np.ndarray:
    """Implementiert dst = alpha * dst_orig + beta * src_orig."""
    if halo_instance.relu_clamp_axpby_f32 is None: return img
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    
    input_buffers_info = _convert_image_to_float_channels(img)
    if not input_buffers_info: return img

    output_buffers_info = []
    for i in range(c):
        out_buf_mv, out_stride_bytes = make_aligned_f32_buffer(w, h, components=1, alignment=64)
        out_buf_mv[:w*h] = input_buffers_info[i][0][:w*h] 
        output_buffers_info.append((out_buf_mv, out_stride_bytes))

    def axpby_wrapper(src, ss, src_w, src_h, dst_buf, dst_stride, **kwargs):
        halo_instance.relu_clamp_axpby_f32(
            src, dst_buf, src_w, src_h, ss, dst_stride, 
            alpha=alpha, beta=beta, clamp_min=clamp_min, clamp_max=clamp_max, 
            apply_relu=apply_relu, use_mt=True
        )

    for i in range(c):
        src_buf, src_stride = input_buffers_info[i]
        dst_buf, dst_stride = output_buffers_info[i]

        try:
            axpby_wrapper(src_buf, src_stride, w, h, dst_buf, dst_stride)
        except Exception as e:
            print(f"HALO Laufzeitfehler in relu_clamp_axpby_f32: {e}")
            return img
    
    return _convert_float_channels_to_image(output_buffers_info, h, w)


# ====================================================================
# ABSCHNITT 3: Gradio UI Definition
# ====================================================================

with gr.Blocks(title="HALO Image Processor Demo") as demo:
    gr.Markdown(
        """
        # HALO High-Performance Image Processor (Demo)
        Demonstriert die Nutzung der hochoptimierten HALO-Bibliothek (C++ mit AVX2/MT) 
        innerhalb einer Gradio-Web-App. **Alle Filter werden pro Kanal (R/G/B) auf float32-Daten angewendet.**
        """
    )
    
    with gr.Row():
        image_input = gr.Image(label="Eingabebild", type="numpy", interactive=True)
        image_output = gr.Image(label="Ergebnisbild (HALO)", type="numpy")

    gr.Markdown("---")
    
    with gr.Tabs():
        
        with gr.TabItem("Weichzeichner / Filter"):
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
                
                btn_box.click(process_blur_box, inputs=[image_input, radius_box], outputs=image_output)
                btn_gauss.click(process_blur_gaussian, inputs=[image_input, sigma_gauss], outputs=image_output)
                btn_sobel.click(process_filter_sobel, inputs=[image_input], outputs=image_output)
                btn_median.click(process_filter_median, inputs=[image_input], outputs=image_output)
                btn_unsharp.click(process_filter_unsharp, inputs=[image_input, sigma_unsharp, amount_unsharp, threshold_unsharp], outputs=image_output)

        with gr.TabItem("Tonwert / Farbe"):
            gr.Markdown("### Helligkeits- und Farbraumkorrekturen (Float-Bereich 0-1)")
            
            with gr.Row():
                btn_invert = gr.Button("Invertieren")
                btn_invert.click(process_tonewheel_invert, inputs=[image_input, gr.State(True)], outputs=image_output)

            with gr.Row():
                gamma_gamma = gr.Slider(label="Gamma", minimum=0.1, maximum=5.0, step=0.1, value=2.2)
                gamma_gain = gr.Slider(label="Gain", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                btn_gamma = gr.Button("Gamma Anwenden")
                btn_gamma.click(process_tonewheel_gamma, inputs=[image_input, gamma_gamma, gamma_gain], outputs=image_output)

            with gr.Row():
                levels_low = gr.Slider(label="Input Low (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.1)
                levels_high = gr.Slider(label="Input High (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.9)
                levels_gamma = gr.Slider(label="Gamma", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                btn_levels = gr.Button("Levels Anwenden")
                btn_levels.click(process_tonewheel_levels, inputs=[image_input, levels_low, levels_high, levels_gamma], outputs=image_output)
                
            with gr.Row():
                threshold_low = gr.Slider(label="Threshold Low (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.4)
                threshold_high = gr.Slider(label="Threshold High (0-1)", minimum=0.0, maximum=1.0, step=0.01, value=0.6)
                btn_threshold = gr.Button("Threshold Anwenden")
                btn_threshold.click(process_tonewheel_threshold, inputs=[image_input, threshold_low, threshold_high], outputs=image_output)

        with gr.TabItem("Morphologie"):
            gr.Markdown("### Morphologische 3x3 Operationen (Zwei-Pass Separabel)")
            
            with gr.Row():
                btn_erode = gr.Button("Erode 3x3 (Erosion)")
                btn_dilate = gr.Button("Dilate 3x3 (Dilatation)")
            with gr.Row():
                btn_open = gr.Button("Open 3x3 (Erode dann Dilate)")
                btn_close = gr.Button("Close 3x3 (Dilate dann Erode)")
                
            btn_erode.click(process_morphology, inputs=[image_input, gr.State("Erode")], outputs=image_output)
            btn_dilate.click(process_morphology, inputs=[image_input, gr.State("Dilate")], outputs=image_output)
            btn_open.click(process_morphology, inputs=[image_input, gr.State("Open")], outputs=image_output)
            btn_close.click(process_morphology, inputs=[image_input, gr.State("Close")], outputs=image_output)


        with gr.TabItem("Geometrie"):
            gr.Markdown("### Geometrische Operationen")
            
            with gr.Row():
                flip_h = gr.Checkbox(label="Horizontal spiegeln", value=True)
                flip_v = gr.Checkbox(label="Vertikal spiegeln", value=False)
                btn_flip = gr.Button("Spiegeln (Flip)")
                btn_flip.click(process_geometry_flip, inputs=[image_input, flip_h, flip_v], outputs=image_output)
                
            with gr.Row():
                # Ändere zu Dropdown, da Radio mit int-Values manchmal Probleme macht
                rotate_turns = gr.Radio(label="90° Drehungen", choices=[1, 2, 3], value=1, type="value")
                btn_rotate = gr.Button("Drehen um 90°/180°/270°")
                btn_rotate.click(process_geometry_rotate, inputs=[image_input, rotate_turns], outputs=image_output)
            
            with gr.Row():
                resize_w = gr.Slider(label="Neue Breite", minimum=64, maximum=1024, step=16, value=512)
                resize_h = gr.Slider(label="Neue Höhe", minimum=64, maximum=1024, step=16, value=512)
                resize_interp = gr.Radio(label="Interpolation", choices=["Bilinear", "Bicubic"], value="Bilinear")
                btn_resize = gr.Button("Bildgröße ändern")
                btn_resize.click(process_geometry_resize, inputs=[image_input, resize_w, resize_h, resize_interp], outputs=image_output)


        with gr.TabItem("Custom: AXPBY + ReLU/Clamp"):
            gr.Markdown(
                """
                ### Erweiterte AXPBY-Operation (`dst = alpha * dst_orig + beta * src_orig`)
                Da Gradio nur ein Bild zulässt, wird das Eingabebild als `src_orig` und `dst_orig` verwendet.
                """
            )
            
            with gr.Row():
                alpha_val = gr.Slider(label="Alpha (Gewichtung dst_orig)", minimum=0.0, maximum=2.0, step=0.01, value=0.0)
                beta_val = gr.Slider(label="Beta (Gewichtung src_orig)", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                
            with gr.Row():
                clamp_min_val = gr.Slider(label="Clamp Min (0.0)", minimum=-0.5, maximum=0.5, step=0.01, value=0.0)
                clamp_max_val = gr.Slider(label="Clamp Max (1.0)", minimum=0.5, maximum=1.5, step=0.01, value=1.0)
                apply_relu_check = gr.Checkbox(label="Apply ReLU (max(0, val))", value=False)
                
            btn_axpby = gr.Button("AXPBY Anwenden")
            btn_axpby.click(process_custom_axpby, 
                             inputs=[image_input, alpha_val, beta_val, clamp_min_val, clamp_max_val, apply_relu_check], 
                             outputs=image_output)

# Starte die Gradio App
if __name__ == "__main__":
    demo.launch()
