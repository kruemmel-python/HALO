# HALO: High-throughput Array and Logic Operations

**HALO** (v0.5b) ist eine **hybride Rechen-Engine** in C++ und OpenCL, die für eine extrem schnelle Verarbeitung von Bild- und Array-Daten entwickelt wurde. HALO überwindet Performance-Grenzen, indem es **handoptimierte SIMD-CPU-Kerne (AVX2)** und **GPU-OpenCL-Fähigkeiten** nahtlos in einem eleganten Python-Wrapper vereint.

---

## ✨ Highlights & Alleinstellungsmerkmale

*   **Dual-Core Beschleunigung:** Nahtloser Fallback oder explizite Nutzung zwischen dedizierten CPU-Fast-Path (AVX2/MT) und OpenCL-GPU-Kernen.
*   **Zero-Copy Python Bridge:** Effiziente Datenübergabe über Python `ctypes` mit Alignment-garantierten, gepinnten Speichern.
*   **Intelligentes Auto-Tuning:** Automatische Kalibrierung und Auswahl der schnellsten CPU-Implementierung zur Initialisierungszeit (Skalar, SSE2, AVX2, Streaming).
*   **Produktionsreife Features:** Enthält hochentwickelte Kernel für Bildverarbeitung (z.B. Bikubische Interpolation, Separable Blur) und GPU-beschleunigte ML-Primitive (MatMul, Adam, LayerNorm).
*   **Fokus auf I/O und Cache:** Reduzierung des Cache-Drucks durch bedingte Verwendung von Non-Temporal (Streaming) Stores bei großen Datenmengen.

---

## 🏗️ Architekturübersicht

Die HALO-Architektur ist in eine schlanke Python-Schicht und eine leistungsstarke native Bibliothek unterteilt.

| Komponente | Sprache | Fokus & Rolle |
| :--- | :--- | :--- |
| **`halo_driver.dll` / `.so`** | C/C++ (Unity Build) | Die native Shared Library, welche beide Beschleunigungspfade vereint. |
| **`fastpath.cpp`** | C++ (SIMD, MT, C++17) | **CPU Fast-Path:** Enthält den persistenten Worker-Thread-Pool, das dynamische Scheduling und alle AVX2/FMA-optimierten numerischen Kernel. |
| **`CipherCore_OpenCl.c`** | C (OpenCL 1.2+) | **GPU-Hybrid-Engine:** Verwaltet OpenCL-Kontext, Kommando-Queues, GPU-Speicher (Buffers) und enthält die Source-Codes für alle GPU-Kernel (z.B. MatMul, Adam, Box-Blur). |
| **`halo.py`** | Python (`ctypes`) | **Binding Layer:** Lädt die DLL, implementiert Autotuning, initialisiert den Thread-Pool und führt die API-Calls aus, inklusive Speicher-Management-Helfern. |
| **`halo_extensions.py`** | Python (`NumPy`) | **Erweiterte Logik:** Stellt anspruchsvolle Algorithmen (Canny, Bilateral) bereit, die auf die f32-Arrays des HALO-Kerns angewendet werden können. |

---

## ⚙️ Installation & Kompilierung

### 1. Voraussetzungen

*   C++17 kompatibler Compiler (z.B. GCC/MinGW-w64, MSVC).
*   **OpenCL SDK:** Notwendig für die Kompilierung des GPU-Treibers.
*   **Python 3.9+**
    ```bash
    pip install numpy gradio # Gradio ist optional für die Demo
    # Optional für experimentelle GPU-Pfad-Tests (CuPy):
    # pip install cupy-cuda12x 
    ```

### 2. Native Kompilierung (`halo_driver.dll`)

Die `halo_driver.dll` muss manuell aus den drei C/C++-Quelldateien gebaut werden. Das folgende Kommando nutzt `g++` und bindet den OpenCL-Treiber dynamisch (`-lOpenCL`).

| System | Kommando |
| :--- | :--- |
| **Windows (MinGW-w64)** | `g++ -std=c++17 -O3 -march=native -shared -o halo_driver.dll halo_driver.cpp -I./CL -L./CL -lOpenCL` |
| **Linux (GCC)** | `g++ -std=c++17 -O3 -march=native -shared -o libhalo_driver.so halo_driver.cpp -I./CL -L./CL -lOpenCL` |
| **MSVC (Windows)** | `cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 halo_driver.cpp /Fe:halo_driver.dll` |

*Stellen Sie sicher, dass die OpenCL-Header und -Bibliotheken über die Pfade `-I` und `-L` erreichbar sind.*

---

## 🚀 Quickstart: Python API

Die Klasse `HALO` verwaltet den gesamten Lebenszyklus und die Konfiguration des nativen Kerns.

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer
from halo_extensions import canny_edge_detector # High-Level Extension

# --- 1. Initialisierung und Konfiguration ---
# Erzwingt 8 Threads und versucht, GPU-Unterstützung zu initialisieren.
halo = HALO(threads=8, use_gpu=True) 

print(f"HALO Version: {halo.HALO_VERSION}")
print(f"CPU Features: {halo.features}")
if halo.gpu_enabled:
    print(f"GPU aktiv: Gerät {halo.gpu_device}")
else:
    print("GPU-Pfad nicht verfügbar, verwende CPU Fast-Path.")

# --- 2. Speicher-Management für C++ Aufrufe ---
W, H = 1024, 1024
# Erstellt einen 64-Byte-ausgerichteten (aligned) Puffer
src_buf_mv, stride = make_aligned_f32_buffer(W, H, components=1) 
data_view = np.frombuffer(src_buf_mv, dtype=np.float32).reshape(H, W) 

# Testdaten generieren (nur zur Demonstration)
data_view[:] = np.random.rand(H, W).astype(np.float32)

# --- 3. Aufruf des C++ Kernels (AVX2/MT) ---
# Führt die Invertierung auf dem CPU-Fast-Path aus.
halo.invert_f32(
    src=src_buf_mv, 
    dst=src_buf_mv, 
    width=W, 
    height=H, 
    src_stride_bytes=stride, 
    dst_stride_bytes=stride,
    use_mt=True
)

# --- 4. Nutzung der High-Level Extension (NumPy / CPU-Fallback) ---
# Die Extension erwartet und liefert NumPy-Arrays (automatische Konvertierung)
input_rgb_uint8 = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

# Führt den Canny Algorithmus aus (läuft in Python/NumPy/CuPy-Fallback)
edges = canny_edge_detector(input_rgb_uint8, high_threshold=0.3)

print(f"\nVerarbeitung abgeschlossen. Ergebnis-Array-Typ: {edges.dtype}")

# Sauberes Beenden des Thread-Pools (auch durch atexit registriert)
halo.close()
```

---

## 📦 Python Modul-API (Auszug)

Die Kern-Operationen sind in der `HALO` Klasse implementiert und akzeptieren Python `array.array` oder `memoryview` Objekte, die den nativen Puffer repräsentieren.

| Kern-Funktion | Beschreibung | Native Implementierung |
| :--- | :--- | :--- |
| `halo.saxpy(a, x, y)` | Berechnet $y = a \cdot x + y$ (Vektor). | `fastpath.cpp` (Autotuned SIMD) |
| `halo.gaussian_blur_f32(...)`| Separabler Gauß-Filter auf f32-Bildern. | `fastpath.cpp` (AVX2/MT) / `CipherCore_OpenCl.c` (OpenCL) |
| `halo.img_u8_to_f32_lut_axpby(...)`| Komplexer Konvertierungskern (u8 zu f32, mit LUT, Scale/Offset und AXPBY). | `fastpath.cpp` (Affiner Fast-Path/Gather) |
| `halo.execute_matmul_on_gpu(...)`| Führt eine Matrizenmultiplikation auf der GPU aus (direkter Aufruf des OpenCL-Kerns). | `CipherCore_OpenCl.c` (OpenCL Kernel) |
| `make_aligned_f32_buffer(...)`| Utility: Erzeugt einen Puffer mit garantierter Speicherausrichtung. | `halo.py` |

---

## 🕹️ Interaktive Demo

Für eine einfache Demonstration der Leistungsfähigkeit aller Funktionen (CPU-Filter, High-Level-Extensions und Video-Verarbeitung) starten Sie die Gradio-Anwendung:

```bash
python halo_demo_app.py
```
Dies startet einen lokalen Webserver, der im Browser zugänglich ist.

---

*© 2025 Ralf Krümmel – Lizenziert unter MIT.*
