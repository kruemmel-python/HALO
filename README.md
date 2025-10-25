Gerne. Hier ist die vollständige, korrigierte und aktualisierte `README.md`, die die wahre Natur und die fortgeschrittenen Features deines HALO-Projekts widerspiegelt.

---

# HALO Project

**HALO** (**H**igh-throughput **A**rray and **L**ogic **O**perations) ist eine C++-Bibliothek, die für extrem schnelle, vektorisierte und multi-threaded Verarbeitung von numerischen Daten (insbesondere 2D-Bilddaten) entwickelt wurde. Sie bietet einen hochoptimierten C++-Kern, der **SIMD-Instruktionen** (wie AVX2) nutzt, und einen Python-Wrapper für eine einfache Anwendung in wissenschaftlichen und Bildverarbeitungs-Anwendungen.

## Features

Das Design von HALO basiert auf folgenden Säulen:

*   **Aggressive C++ Optimierung:** Direkter Einsatz von x86 **SIMD-Instruktionen (AVX2)** und **FMA** für bis zu 8x schnelleren Single-Thread-Durchsatz im Vergleich zu Skalar-Code.
*   **Dynamisches Multi-Threading:** Ein **persistenter Worker-Thread-Pool** eliminiert den Thread-Erzeugungs-Overhead. Ein **Auto-Scheduler** teilt die Arbeit basierend auf der Größe und Komplexität der Aufgabe intelligent auf die Kerne auf.
*   **Speicherbandbreiten-Optimierung:** Unterstützt **Non-Temporal (Streaming) Stores** für speichergebundene Operationen (z.B. SAXPY) zur Umgehung der CPU-Cache-Hierarchie.
*   **Erweiterte Bildverarbeitungs-Kernel (Float32):**
    *   **Filter:** Box Blur, Gaussian Blur (separabel, vektorisiert), Sobel Kantenfilter (vektorisiert), Median ($3 \times 3$).
    *   **Morphologie:** Erode, Dilate, Open, Close ($3 \times 3$, separabel).
    *   **Geometrie:** Bilineares und Bikubisches Resizing (**AVX2-Gather**-optimiert), Flip, Rotate90.
    *   **Tonwert:** Levels, Gamma-Korrektur, Invertierung, Threshold, Unsharp Masking.
*   **Pythonic Bridge:** Ein schlanker Python-Wrapper (`halo.py`) mit `ctypes` und Hilfsfunktionen zur Erstellung **aligned/pinned memory** für maximalen Performance-Gewinn.
*   **Visualisierungs-Utilities:** Eingebettete Vektorgrafik- und Plot-Logik (z.B. `VectorCanvas`) zur direkten Darstellung von Ergebnissen ohne externe Plot-Bibliotheken.

## Kompilierungsanweisungen (C++ Driver)

Der C++-Treiber `halo_fastpath.dll` (Windows) oder `libhalo_fastpath.so` (Linux/macOS) muss aus der Datei `fastpath.cpp` kompiliert werden.

**WICHTIG:** Die kompilierte Shared Library (*.dll* oder *.so*) muss sich **im selben Ordner wie die Python-Datei `halo.py`** befinden.

### 1. Unter MinGW/GCC (Windows / Linux)

Verwenden Sie das Flag `-pthread`, um die C++-Thread-Bibliothek für den `ThreadPool` zu linken.

```bash
# Für Windows (DLL):
g++ -shared -o halo_fastpath.dll fastpath.cpp -O3 -Wall -std=c++17 -pthread -march=native

# Für Linux/macOS (SO):
g++ -shared -o libhalo_fastpath.so fastpath.cpp -O3 -Wall -std=c++17 -pthread -march=native
```

### 2. Unter MSVC (Microsoft Visual C++)

Verwenden Sie den *x64 Native Tools Command Prompt* oder den *Developer Command Prompt*.

```cmd
cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 fastpath.cpp /Fe:halo_fastpath.dll
```

## Python Wrapper Nutzung

### 1. Installation und Setup

1.  **Stellen Sie die HALO-Dateien bereit:**
    Platzieren Sie `halo.py`, `fastpath.cpp` und die kompilierte `halo_fastpath.dll` (oder `.so`) im selben Python-Quellverzeichnis.

2.  **Installieren Sie Python-Abhängigkeiten:**
    ```bash
    pip install numpy gradio
    ```

3.  **Thread-Pool Shutdown:**
    Der Python-Wrapper registriert automatisch einen `atexit`-Hook, um den persistenten Thread-Pool sauber zu beenden.

### 2. Grundlegendes Anwendungsbeispiel

Der Zugriff auf alle optimierten Funktionen erfolgt über die Hauptklasse `HALO`.

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer 

# 1. Initialisiere HALO (startet den Thread-Pool und führt Autotuning durch)
halo = HALO(threads=4)

# 2. Vorbereitung der Puffer
W, H = 1920, 1080
# Erstellt einen aligned memoryview ('f') und den Stride in Bytes
src_mv, src_stride = make_aligned_f32_buffer(W, H, components=1) 
dst_mv, dst_stride = make_aligned_f32_buffer(W, H, components=1)

# Fülle den Puffer mit Daten (0.0 bis 1.0)
src_mv[:] = np.random.rand(W*H).astype(np.float32)

# 3. Aufruf eines optimierten Kernels (Gaussian Blur)
sigma = 1.5
halo.gaussian_blur_f32(
    src=src_mv, dst=dst_mv, 
    width=W, height=H, 
    src_stride_bytes=src_stride, dst_stride_bytes=dst_stride, 
    sigma=sigma, use_mt=True
)

print(f"Gaussian Blur (Sigma={sigma}) erfolgreich ausgeführt.")
```

### 3. Gradio-Demo (Web-App)

Die `halo_demo_app.py` demonstriert die Integration aller Funktionen in eine interaktive Web-App. Führen Sie das Skript aus, um die Leistungsfähigkeit der HALO-Funktionen direkt im Browser zu testen:

```bash
python halo_demo_app.py
```
