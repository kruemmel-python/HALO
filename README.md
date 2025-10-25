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

Der C++-Treiber muss aus der Datei `fastpath.cpp` kompiliert werden.

**WICHTIG:** Die kompilierte Shared Library (*.dll* oder *.so*) muss sich **im selben Ordner wie die Python-Datei `halo.py`** befinden.

### 1. Unter MinGW/GCC (Windows / Linux)

Verwenden Sie die bewährte Kommando-Struktur, um die beste Kompatibilität und Leistung zu gewährleisten. Das Flag **`-pthread`** ist für die Thread-Funktionalität erforderlich.

```bash
# Für Windows (DLL):
g++ -O3 -march=native -pthread -shared -o halo_fastpath.dll fastpath.cpp

# Für Linux/macOS (SO):
g++ -O3 -march=native -pthread -shared -o libhalo_fastpath.so fastpath.cpp
```

| Flag | Bedeutung |
| :--- | :--- |
| `-O3` | Aggressive Compiler-Optimierung. |
| `-march=native`| Erzeugt Code für die native CPU-Architektur (wichtig für AVX2/FMA). |
| `-pthread` | Bindet die POSIX Threads Library ein (für `std::thread`). |
| `-shared` | Erstellt eine Shared Library (DLL/SO). |

### 2. Unter MSVC (Microsoft Visual C++)

Verwenden Sie den *x64 Native Tools Command Prompt* oder den *Developer Command Prompt*.

```cmd
cl /LD /EHsc /O2 /arch:AVX2 /std:c++17 fastpath.cpp /Fe:halo_fastpath.dll
```

| Flag | Bedeutung |
| :--- | :--- |
| `/LD` | Erstellt eine DLL. |
| `/O2` | Optimierungslevel 2 (Geschwindigkeit). |
| `/arch:AVX2`| Aktiviert AVX2 SIMD-Instruktionen. |
| `/std:c++17`| Verwendet den C++17-Standard. |

## Python Wrapper Nutzung

### 1. Installation und Setup

1.  **Stellen Sie die HALO-Dateien bereit:**
    Platzieren Sie `halo.py` und die kompilierte Library (`*.dll` oder `*.so`) im selben Python-Quellverzeichnis.

2.  **Installieren Sie Python-Abhängigkeiten:**
    ```bash
    pip install numpy gradio
    ```

### 2. Grundlegendes Anwendungsbeispiel

Der Zugriff auf alle optimierten Funktionen erfolgt über die Hauptklasse `HALO`.

```python
import numpy as np
from halo import HALO, make_aligned_f32_buffer 
import math # Für Sigma-Berechnung

# 1. Initialisiere HALO (startet den Thread-Pool und führt Autotuning durch)
halo = HALO(threads=4)

# 2. Vorbereitung der Puffer (1920x1080)
W, H = 1920, 1080

# Erstellt einen aligned memoryview ('f') und den Stride in Bytes
src_mv, src_stride = make_aligned_f32_buffer(W, H, components=1) 
dst_mv, dst_stride = make_aligned_f32_buffer(W, H, components=1)

# Fülle den Puffer mit Beispieldaten (z.B. aus einem NumPy Array)
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

### 3. Interaktive Gradio-Demo (Web-App)

Führen Sie das Gradio-Demo-Skript aus, um alle HALO-Funktionen interaktiv zu testen:

```bash
python halo_demo_app.py
```
