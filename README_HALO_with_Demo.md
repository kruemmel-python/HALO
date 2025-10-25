# ⚡ HALO: High-throughput Array and Logic Operations

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17%2B-orange.svg)
![OpenCL](https://img.shields.io/badge/OpenCL-1.2%2B-lightgrey.svg)
![Version](https://img.shields.io/badge/version-0.5b-orange.svg)

**HALO** ist eine **hybride Rechen-Engine** in C++ und OpenCL, die für extrem schnelle Verarbeitung von Bild- und Array-Daten entwickelt wurde.  
Sie vereint **handoptimierte SIMD-CPU-Kerne (AVX2)** und **GPU-OpenCL-Fähigkeiten** in einem eleganten Python-Wrapper.

---

## ✨ Highlights & Alleinstellungsmerkmale

- 🚀 **Dual-Core Beschleunigung:** Nahtloser Wechsel zwischen CPU-Fast-Path (AVX2/MT) und OpenCL-GPU-Kernen  
- 🧠 **Intelligentes Auto-Tuning:** Automatische Kalibrierung zur Laufzeit (Skalar, SSE2, AVX2, Streaming)
- 🧩 **Zero-Copy Bridge:** Direkter Zugriff auf native Speicher (Alignment & Pinned Memory)
- 🧮 **Produktionsreife Kernel:** Bildverarbeitung, MatMul, LayerNorm, Adam, BoxBlur u. v. m.
- ⚙️ **Optimiertes I/O-Design:** Cache-aware Memory-Zugriff, Non-Temporal Stores für große Datenmengen

---

## 🏗️ Architekturübersicht

| Komponente | Sprache | Fokus & Rolle |
|:------------|:---------|:--------------|
| **`halo_driver.dll` / `.so`** | C/C++ | Native Shared Library – vereint CPU & GPU Beschleunigung |
| **`fastpath.cpp`** | C++ (SIMD, MT) | CPU Fast-Path mit Thread-Pool & AVX2/FMA-Kernen |
| **`CipherCore_OpenCl.c`** | C (OpenCL 1.2+) | GPU-Engine: OpenCL-Kontext, Buffers, Kernel (MatMul, Adam, BoxBlur) |
| **`halo.py`** | Python (`ctypes`) | Binding Layer – DLL-Lader, Thread-Init, API-Wrapper |
| **`halo_extensions.py`** | Python (`NumPy`) | High-Level Extensions (Canny, Bilateral etc.) |

---

## ⚙️ Installation & Kompilierung

### Voraussetzungen

- C++17-kompatibler Compiler (MSVC / GCC / MinGW-w64)
- OpenCL SDK (z. B. AMD APP SDK, Intel OneAPI, NVIDIA CUDA Toolkit)
- Python 3.9 oder höher

```bash
pip install numpy gradio
# Optional für GPU-Fallbacks:
# pip install cupy-cuda12x
```

### Kompilierung der nativen Bibliothek

| System | Kommando |
|:--------|:----------|
| **Windows (MinGW-w64)** | `g++ -std=c++17 -O3 -march=native -shared -o halo_driver.dll halo_driver.cpp -I./CL -L./CL -lOpenCL` |
| **Linux (GCC)** | `g++ -std=c++17 -O3 -march=native -shared -o libhalo_driver.so halo_driver.cpp -I./CL -L./CL -lOpenCL` |
| **MSVC (Windows)** | `cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 halo_driver.cpp /Fe:halo_driver.dll` |

> **Hinweis:** Stelle sicher, dass die OpenCL-Headers und -Libs im Pfad `-I` / `-L` verfügbar sind.

---

## 🚀 Quickstart (Python)

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer
from halo_extensions import canny_edge_detector

halo = HALO(threads=8, use_gpu=True)
print("CPU Features:", halo.features)

# Puffer erstellen
W, H = 1024, 1024
src_buf, stride = make_aligned_f32_buffer(W, H)
arr = np.frombuffer(src_buf, dtype=np.float32).reshape(H, W)
arr[:] = np.random.rand(H, W).astype(np.float32)

# CPU-Invertierung (AVX2)
halo.invert_f32(src_buf, src_buf, W, H, stride, stride, use_mt=True)

# High-Level Extension
rgb = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
edges = canny_edge_detector(rgb, high_threshold=0.3)
print("Edges:", edges.shape)

halo.close()
```

---

## 🧩 Python Modul-API (Auszug)

| Funktion | Beschreibung | Implementierung |
|:----------|:--------------|:----------------|
| `saxpy(a, x, y)` | $y = a \cdot x + y$ (SIMD) | `fastpath.cpp` |
| `gaussian_blur_f32(...)` | Separable Gauß-Filter | `fastpath.cpp` / `CipherCore_OpenCl.c` |
| `img_u8_to_f32_lut_axpby(...)` | LUT & Konvertierung u8→f32 | `fastpath.cpp` |
| `execute_matmul_on_gpu(...)` | GPU-Matrixmultiplikation | `CipherCore_OpenCl.c` |
| `make_aligned_f32_buffer(...)` | Speicher-Helfer (64B-aligned) | `halo.py` |

---

## 🕹️ Interaktive Demo

Für die interaktive GUI:

```bash
python halo_demo_app.py
```

👉 Öffne danach im Browser: [http://localhost:7860](http://localhost:7860)

**Demo-Features:**
- Echtzeit-Filter (Box, Sobel, Gaussian)
- Resize, Invert, Bilateral
- GPU/CPU Auswahl per Sidebar
- Performance-Timings (ms)

---

## 📸 Screenshots & GUI-Vorschau

| CPU Fast-Path (AVX2) | GPU OpenCL Hybrid |
|:---------------------:|:----------------:|
| ![CPU Fast-Path](docs/images/cpu_fastpath.png) | ![GPU Hybrid](docs/images/gpu_hybrid.png) |

> Screenshots liegen unter `docs/images/`  
> Du kannst sie aus der Demo-App exportieren oder eigene Benchmarks dort ablegen.

---

## 💾 Download & Repository-Struktur

**Download ZIP:**  
👉 [HALO v0.5b (Release-Paket)](https://github.com/kruemmel-python/HALO/releases/latest)

**Ordnerübersicht:**
```
HALO/
├─ src/
│  ├─ fastpath.cpp
│  ├─ CipherCore_OpenCl.c
│  ├─ halo_driver.cpp
│
├─ python/
│  ├─ halo.py
│  ├─ halo_extensions.py
│
├─ demo/
│  └─ halo_demo_app.py
│
├─ docs/
│  ├─ images/
│  └─ benchmarks/
│
└─ README.md
```

---

## 🧪 Benchmark-Vorschau

| Operation | Größe | CPU (AVX2) | GPU (RX 6500M) |
|:-----------|:------|:------------|:----------------|
| `MatMul f32` | 1024×1024 | 85 ms | 6.2 ms |
| `GaussianBlur f32` | 2048×2048 | 64 ms | 9.7 ms |
| `Invert f32` | 4096×4096 | 12 ms | 3.8 ms |

> Getestet auf AMD Ryzen 7 7735HS + Radeon RX 6500M  
> (HALO v0.5b Build ID: 2025-10-25)

---


---

## ⚡ Performance Showcase (HALO v0.5b)

| Operation | Pfad | Zeit | Beschreibung |
|:-----------|:------|:------|:--------------|
| **Unsharp Masking** | C++ AVX2 Fast-Path | **< 1 s** | Nahezu Echtzeit – Multi-Threaded SIMD-Verarbeitung |
| **Canny Edge Detector** | Python / NumPy | **10.9 s** | Präzise Kantendetektion mit adaptivem Schwellwert |
| **Bilateral Filter** | Python / NumPy | **22.5 s** | Rechenintensive Glättung mit Erhalt lokaler Strukturen |
| **Morphology (Blackhat)** | Python / NumPy | **11.8 s** | Erweiterte Strukturfilterung im f32-Raum |

### Beispielansichten (aus der HALO-Demo-App)

| Unsharp Masking (AVX2) | Geometrie (Flip + Rotate) |
|:----------------------:|:------------------------:|
| ![Unsharp Masking](docs/images/unsharp_mask_demo.png) | ![Geometrie](docs/images/geometry_flip_demo.png) |

| AXBPY / ReLU Clamp | High-Level Morphologie |
|:------------------:|:----------------------:|
| ![AXBPY ReLU](docs/images/axpby_relu_demo.png) | ![Morphology](docs/images/morphology_demo.png) |

> Alle Tests auf AMD Ryzen 7 7735HS + Radeon RX 6500M  
> (HALO v0.5b Build ID 2025-10-25, Python 3.12)


## 🧠 Philosophie & Design

> „HALO ist keine bloße Bibliothek – es ist ein Konzept:  
> rohe C++-Leistung, in Einklang gebracht mit der Eleganz von Python.“  
> — *Ralf Krümmel, Entwickler von HALO*

HALO verbindet **maschinennahes Engineering** (SIMD, Cache-Alignment, OpenCL)  
mit **moderner Software-Architektur** (Python-Bindings, NumPy-Integration).  
Das Ergebnis: eine Engine, die die Grenzen von High-Level und Low-Level nahtlos verschmelzen lässt.

---

## 📜 Lizenz & Autor

**Autor:** [Ralf Krümmel](https://www.linkedin.com/in/ralf-krümmel-3b6250335)  
**Lizenz:** MIT  
**© 2025 Ralf Krümmel**
