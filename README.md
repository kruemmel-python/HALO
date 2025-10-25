# HALO Project

**HALO** (**H**igh-throughput **A**rray and **L**ogic **O**perations) ist eine hochoptimierte Softwarelösung für die rechenintensive numerische und Bilddatenverarbeitung. Das Projekt kombiniert einen performanten C++-Kern, der Hardware-Optimierungen wie SIMD (AVX2/FMA) und Multi-Threading nutzt, mit einem schlanken Python-Wrapper für eine nahtlose Anwendungsintegration.

*Version: 1.0 (2025-10-25)*

## 2. Executive Summary

HALO wurde entwickelt, um die Performance-Grenzen typischer Python-basierter Bibliotheken bei der Verarbeitung großer Array-Daten zu überwinden. Der Kern fokusiert auf **aggressive, hardwarenahe Optimierungen** wie die direkte Nutzung von **AVX2-SIMD**-Instruktionen und einem **intelligenten Multi-Threading-Modell** (persistenter Thread-Pool mit Auto-Scheduler).

HALO richtet sich an Ingenieure, Datenwissenschaftler und Forscher in Bereichen wie **Bildverarbeitung, Computer Vision** und **wissenschaftliche Simulationen**, die eine beispiellose Geschwindigkeit und Speichereffizienz bei pixelweisen Operationen, Filtern oder geometrischen Transformationen benötigen.

## 3. Kern-Features und Optimierungen

| Kategorie | Merkmale | Hardware-Optimierungen |
| :--- | :--- | :--- |
| **Performance-Kern** | C++ Shared Library (`fastpath.cpp`) für maximale Ausführungsgeschwindigkeit. | Direkter Einsatz von **AVX2/FMA** SIMD-Instruktionen. |
| **Multi-Threading**| **Persistenter Thread-Pool** eliminiert Erzeugungs-Overhead. | **Dynamischer Auto-Scheduler** für optimale Lastverteilung (geringere Latenz). |
| **Speicher-I/O** | Spezielle Algorithmen für speichergebundene Operationen (z.B. SAXPY). | **Non-Temporal (Streaming) Stores** reduzieren Cache-Druck. |
| **Funktionalität** | Breite Palette an Kernels: Blur, Sobel, Median, Erode/Dilate, Levels, Gamma, Invert, Resize (Bilinear/Bicubic mit AVX2-Gather). | Python-Wrapper unterstützt **Aligned/Pinned Memory** für Zero-Copy-Datenübergabe. |

## 4. Kompilierungsanweisungen (C++ Driver)

Die C++-Bibliothek muss kompiliert werden, um die Hardware-spezifischen Optimierungen zu nutzen.

**WICHTIG:** Die kompilierte Shared Library (`halo_fastpath.dll` oder `libhalo_fastpath.so`) muss sich **im selben Ordner wie die Python-Datei `halo.py`** befinden.

### 4.1. Unter MinGW/GCC (Windows / Linux)

Verwenden Sie das bewährte Kommando, das die Optimierungs-, Architektur- und Thread-Flags in die korrekte Reihenfolge bringt:

```bash
# Für Windows (DLL) oder Linux (SO)
g++ -O3 -march=native -pthread -shared -o halo_fastpath.dll fastpath.cpp
# Für Linux/macOS verwenden Sie: -o libhalo_fastpath.so
```

| Flag | Zweck |
| :--- | :--- |
| `-O3` | Aggressive Compiler-Optimierung. |
| `-march=native`| Optimiert für die CPU, auf der kompiliert wird (AVX2/FMA). |
| `-pthread` | Bindet die POSIX Threads Library ein (für `std::thread`). |

### 4.2. Unter MSVC (Microsoft Visual C++)

Verwenden Sie den *x64 Native Tools Command Prompt* oder den *Developer Command Prompt*.

```cmd
cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 fastpath.cpp /Fe:halo_fastpath.dll
```

## 5. Python Wrapper Nutzung

### 5.1. Setup und Beispiel

1.  **Installieren Sie Python-Abhängigkeiten:** `pip install numpy gradio`
2.  **Verwenden Sie die HALO-Klasse:**

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer 

# 1. Initialisiere HALO (Thread-Pool-Start und Autotuning)
halo = HALO(threads=4)

# 2. Erstellung Aligned Memory Puffer (Wichtig für AVX2/Streaming!)
W, H = 1920, 1080
src_mv, src_stride = make_aligned_f32_buffer(W, H, components=1) 
dst_mv, dst_stride = make_aligned_f32_buffer(W, H, components=1)

# Daten füllen... (z.B. aus NumPy)
src_mv[:] = np.random.rand(W*H).astype(np.float32)

# 3. Aufruf eines optimierten Kernels (Gaussian Blur)
sigma = 1.5
halo.gaussian_blur_f32(
    src=src_mv, dst=dst_mv, 
    width=W, height=H, 
    src_stride_bytes=src_stride, dst_stride_bytes=dst_stride, 
    sigma=sigma, use_mt=True
)

print(f"HALO-Kernel erfolgreich ausgeführt. Thread-Pool-Status: {halo.profile.get('cfg', {}).get('threads')} Threads.")
```

### 5.2. Interaktive Demo

Die Gradio-Demo-Anwendung (`halo_demo_app.py`) demonstriert alle Funktionen in einer interaktiven Web-App:

```bash
python halo_demo_app.py
```

