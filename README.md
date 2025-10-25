# HALO Project

**HALO** (**H**igh-throughput **A**rray and **L**ogic **O**perations) ist eine hochoptimierte C++-Bibliothek für array-basierte numerische und Bilddatenverarbeitung, die die rohe Leistung von C++ (SIMD, Multi-Threading) mit der Agilität von Python verbindet. Es wurde entwickelt, um Performance-Engpässe in Anwendungen zu überwinden, die Echtzeitverarbeitung und die Handhabung großer Bildvolumina erfordern.

*Veröffentlicht: 2025-10-25 / Autor: Ralf Krümmel*

---

## 1. HALO-Architektur und Kern-Optimierungen

Das Herzstück von HALO ist ein C++-Kern (`fastpath.cpp`), der für maximale Hardware-Auslastung auf modernen x86-Prozessoren optimiert ist.

### 1.1. Core-Features und C++-Optimierung

| Kategorie | Implementierung | Nutzen & Performance |
| :--- | :--- | :--- |
| **SIMD Vektorisierung** | Direkter Einsatz von **AVX2** und **FMA** (Fused Multiply-Add). | Bis zu **8x Geschwindigkeit** gegenüber Skalar-Code für Kern-Operationen. |
| **Multi-Threading (MT)** | **Persistenter Worker-Thread-Pool** mit intelligentem **Auto-Scheduler**. | Eliminiert Thread-Spawning-Overhead. Nahezu **lineare Skalierung** bei parallelisierbaren Aufgaben. |
| **Speicher I/O** | Unterstützung für **Non-Temporal (Streaming) Stores** (z.B. für SAXPY). | Reduziert **Cache-Druck** und optimiert den Speicherdurchsatz. |
| **Erweiterte Optimierung** | Affine-LUT **Fast-Path** und AVX2-**Gather**-Instruktionen. | Beschleunigt Look-Up-Tabellen und komplexe Interpolationen (Bikubisch/Bilinear). |
| **Python Bridge** | Schlanker **`ctypes`**-Wrapper (`halo.py`) mit Autotuning-Funktionalität. | Ermöglicht **Aligned/Pinned Memory** für Zero-Copy-Datenübergabe. |

### 1.2. Funktionale Übersicht (C++ Kern)

| Kernel-Typ | Verfügbare Operationen |
| :--- | :--- |
| **Filter & Reduktion** | SAXPY, SUM, Box Blur, Gaussian Blur (separabel), Sobel, Median ($3 \times 3$), Unsharp Masking. |
| **Tonwert** | Invert, Gamma-Korrektur, Levels, Threshold, ReLU/Clamp-AXPBY. |
| **Geometrie** | Flip, Rotate90, Resize (Bilinear & Bikubisch). |
| **Morphologie** | Erode, Dilate, Open, Close ($3 \times 3$, separabel). |

---

## 2. High-Level Erweiterungen (Python/NumPy)

Das Modul `halo_extensions.py` nutzt NumPy (optional CuPy) für komplexere Operationen, die nahtlos auf den C++-Grundlagen aufbauen.

| Modul | Features | Datentypen |
| :--- | :--- | :--- |
| **`halo_extensions.py`** | **Bilateral Filter**, **Canny Edge Detector**. Erweiterte Morphologie (Gradient, Top-Hat). | Unterstützt **uint8, uint16, float32, float64** (automatische Normalisierung). |
| **`halo_extensions.py`** | **Farbraum-Konvertierungen** (RGB, HSV, YCbCr, Gray). Affine/Perspektiv-**Warping** (`AffineTransform`). | |
| **`halo_gpu.py`** | Experimentelle GPU-Bridge (SAXPY, Sum, Convolve). | Nutzt **CuPy (CUDA)** als Backend, fällt auf NumPy zurück. |

---

## 3. Kompilierungsanweisungen (C++ Driver)

Die kompilierte Shared Library (`halo_fastpath.dll` oder `libhalo_fastpath.so`) muss sich **im selben Ordner wie `halo.py`** befinden.

### 3.1. Unter MinGW/GCC (Windows / Linux)

Verwenden Sie das Kommando, das die Thread- und Architektur-Flags korrekt anordnet, um Linker-Fehler zu vermeiden:

```bash
# Empfohlenes Kommando
g++ -O3 -march=native -pthread -shared -o halo_fastpath.dll fastpath.cpp
```
*(Verwenden Sie `-o libhalo_fastpath.so` für Linux/macOS.)*

### 3.2. Unter MSVC (Microsoft Visual C++)

Führen Sie den Befehl im *x64 Native Tools Command Prompt* aus:

```cmd
cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 fastpath.cpp /Fe:halo_fastpath.dll
```

---

## 4. Python Wrapper Nutzung

### 4.1. Setup und Installation

1.  **Stellen Sie die HALO-Dateien bereit:**
    `halo.py`, `halo_extensions.py`, `halo_gpu.py` und die kompilierte Library (`*.dll` oder `*.so`) in das Quellverzeichnis kopieren.

2.  **Installieren Sie Python-Abhängigkeiten:**
    ```bash
    pip install numpy gradio
    # Optional für GPU-Funktionalität:
    pip install cupy-cuda12x 
    ```

### 4.2. Grundlegendes Beispiel

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer 
from halo_extensions import bilateral_filter # High-Level Funktion

# 1. Initialisiere HALO (startet den Thread-Pool)
halo = HALO(threads=4)

# 2. Verwenden des C++-Kerns (speicheroptimiert)
W, H = 1024, 1024
src_mv, src_stride = make_aligned_f32_buffer(W, H, components=1) 
src_mv[:] = np.random.rand(W*H).astype(np.float32)

halo.gaussian_blur_f32(
    src=src_mv, dst=src_mv, width=W, height=H, 
    src_stride_bytes=src_stride, dst_stride_bytes=src_stride, 
    sigma=1.0, use_mt=True
)

# 3. Verwenden der High-Level Extension (NumPy-basiert, Dtype-kompatibel)
img_uint8 = np.zeros((H, W, 3), dtype=np.uint8)
img_final = bilateral_filter(img_uint8, diameter=5, sigma_color=0.1)

print("HALO-Verarbeitung erfolgreich ausgeführt.")
```

### 4.3. Interaktive Demo

Führen Sie das Gradio-Demo-Skript aus, um alle C++-optimierten und High-Level-Funktionen interaktiv im Browser zu testen:

```bash
python halo_demo_app.py
```
