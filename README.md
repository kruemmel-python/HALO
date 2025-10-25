# HALO: High-throughput Array and Logic Operations (v0.5b)

**HALO** ist eine hybride C++/OpenCL-Bibliothek für Hochleistungs-Bildverarbeitung und numerische Operationen. Sie kombiniert handoptimierte CPU-Routinen (AVX2, Multi-Threading) mit einem generischen OpenCL-Backend für GPU-Beschleunigung, nahtlos integriert in Python.

*Version: 0.5b / Autor: Ralf Krümmel*

---

## 1. Architektur & Kern-Features

HALO v0.5b verfolgt einen hybriden Ansatz, bei dem eine einzige dynamische Bibliothek (`halo_driver.dll` / `.so`) sowohl die CPU- als auch die GPU-Pfade verwaltet.

### 1.1. Hybrid-Engine (`halo_driver.dll`)

Die Treiber-DLL vereint zwei spezialisierte Kerne:

*   **Fast-Path CPU (`fastpath.cpp`):**
    *   **SIMD-optimiert:** Nutzt AVX2/FMA für maximale Vektorleistung auf modernen CPUs.
    *   **Persistenter Thread-Pool:** Minimiert Overhead durch Wiederverwendung von Worker-Threads.
    *   **Auto-Tuning:** Wählt zur Laufzeit die schnellste Implementierung (Skalar, SSE2, AVX2, AVX2-Streaming) basierend auf der Hardware.
*   **CipherCore GPU (`CipherCore_OpenCl.c`):**
    *   **OpenCL-Backend:** Ermöglicht Hardwarebeschleunigung auf einer Vielzahl von GPUs (NVIDIA, AMD, Intel).
    *   **Just-in-Time Kompilierung:** OpenCL-Kernel werden zur Laufzeit für das spezifische Gerät optimiert kompiliert.
    *   **Breite Palette:** Von einfachen Bildfiltern bis hin zu komplexen ML-Primitiven (Matmul, Adam Optimizer, Embedding Lookups).

### 1.2. Python-Integration

*   **`halo.py`**: Der Low-Level Wrapper. Er lädt die DLL, verwaltet Speicher (Aligned/Pinned Buffers) und entscheidet transparent oder explizit, ob CPU- oder GPU-Pfade genutzt werden. Enthält zudem `atexit`-Hooks für sauberes Ressourcen-Management.
*   **`halo_extensions.py`**: High-Level NumPy-Implementierungen für komplexe Operationen, die (noch) nicht im C++-Kern sind (z.B. Canny Edge, Bilateral Filter, Warping).
*   **`halo_demo_app.py`**: Eine umfassende **Gradio**-Weboberfläche zur Demonstration aller Features, inklusive Echtzeit-Videoverarbeitung und sicherem Temp-File-Management.

---

## 2. Funktionsumfang

| Kategorie | CPU (Fast-Path) | GPU (OpenCL) | Python (Extensions) |
| :--- | :--- | :--- | :--- |
| **Basis-Ops** | SAXPY, Sum (Reduction) | Add, Mul, Clone | - |
| **Bildfilter** | Box, Gaussian, Sobel, Median (3x3), Unsharp Mask | Box, Gaussian, Sobel, Median, Unsharp Mask | Bilateral Filter |
| **Morphologie**| Erode, Dilate, Open, Close (3x3) | - | Erweiterte Morphologie (Gradient, Tophat, etc.) |
| **Tonwert** | Invert, Gamma, Levels, Threshold, ReLU/Clamp | Invert, Gamma, Levels, Threshold | - |
| **Geometrie** | Flip, Rotate90, Resize (Bilinear/Bicubic) | - | Affine & Perspective Warping |
| **ML / Tensor**| - | MatMul (Batched), Softmax/LogSoftmax, GELU, LayerNorm, Adam, Embeddings, CrossEntropy | - |
| **Farbe** | RGB Interleaved/Planar $\leftrightarrow$ f32 | - | RGB $\leftrightarrow$ HSV/Gray/YCbCr |

---

## 3. Build & Installation

### 3.1. Voraussetzungen
*   **Compiler:** C++17 kompatibel (GCC/MinGW-w64 oder MSVC).
*   **OpenCL SDK:** Header (`CL/cl.h`) und Import-Library (z.B. `OpenCL.lib` oder `-lOpenCL`).
*   **Python:** 3.9+ mit `numpy` und `gradio` (für die Demo).

### 3.2. Kompilierung (`halo_driver.dll`)

Da `halo_driver.cpp` die anderen Quellcodedateien inkludiert (Unity-Build), reicht es oft, nur diese Datei zu kompilieren.

**MinGW-w64 (Windows) / GCC (Linux):**
```bash
g++ -std=c++17 -O3 -march=native -shared -o halo_driver.dll halo_driver.cpp -lOpenCL -I./CL -L./CL
# Unter Linux stattdessen: -o libhalo_driver.so
```
*Hinweis: Passen Sie `-I` und `-L` an den Pfad Ihres OpenCL-SDKs an, falls nicht im Systempfad.*

---

## 4. Nutzung

### 4.1. Python API Basic

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer

# Initialisierung (startet CPU-Pool, prüft auf GPU)
halo = HALO(threads=8, use_gpu=True)

if halo.gpu_enabled:
    print(f"GPU aktiv auf Gerät {halo.gpu_device}")

# Speicher erstellen (64-byte aligned für AVX2)
width, height = 1920, 1080
buf, stride = make_aligned_f32_buffer(width, height)
data = np.frombuffer(buf, dtype=np.float32).reshape(height, width) # View als NumPy Array

# Beispiel: Gaussian Blur (wählt automatisch GPU wenn verfügbar & implementiert)
halo.gaussian_blur_f32(buf, buf, width, height, stride, stride, sigma=2.5)
```

### 4.2. Demo-Applikation

Die Gradio-App bietet eine grafische Oberfläche für fast alle Funktionen:

```bash
python halo_demo_app.py
```
*   **Features der Demo:**
    *   Bildverarbeitung mit Vorher/Nachher-Vergleich.
    *   Umschaltbar zwischen C++-Filtern und Python-Extensions.
    *   **Video-Pipeline:** Lädt Videos hoch, verarbeitet sie Frame-by-Frame (z.B. Canny) und speichert das Ergebnis.
    *   Sicheres Temp-Management im `./_tmp` Ordner.

---

## 5. Projektstruktur

*   **Core (C/C++):**
    *   `halo_driver.cpp`: Haupt-Einstiegspunkt, bündelt CPU & GPU Implementierung.
    *   `fastpath.cpp`: CPU-Implementierung (AVX2, Threading).
    *   `CipherCore_OpenCl.c`: OpenCL-Kernel und GPU-Management.
*   **Python Bindings:**
    *   `halo.py`: Haupt-Wrapper Klasse `HALO`.
    *   `halo_extensions.py`: Zusätzliche High-Level Algorithmen (NumPy).
    *   `halo_gpu.py`: Optionaler CuPy-basierter Wrapper (experimentell).
*   **Apps:**
    *   `halo_demo_app.py`: Interaktive Gradio Web-UI.
