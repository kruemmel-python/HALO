# HALO Treiber kompilieren

Diese Anleitung beschreibt Schritt für Schritt, wie Sie die C++-Treiber von HALO kompilieren:

* **CPU Fast-Path** (`fastpath.cpp`) – liefert die AVX2-beschleunigten CPU-Funktionen.
* **Kombinierter GPU-Treiber** (`halo_driver.cpp`) – bündelt den CPU Fast-Path mit dem OpenCL-Treiber (`CipherCore_OpenCl.c`).

Alle Befehle erzeugen Shared Libraries, die anschließend von `halo.py` geladen werden.

---

## 1. Voraussetzungen

| Kategorie | Details |
| :-- | :-- |
| Compiler | **GCC** ≥ 11, **Clang** ≥ 13 oder **MSVC** ≥ 19.3 (Visual Studio 2022). |
| Python | Python 3.9+ für die Demo/Bindings (optional, aber empfohlen). |
| OpenCL | Für den GPU-Treiber: installierter OpenCL ICD + Header/Libraries (z. B. *Intel oneAPI*, *NVIDIA CUDA Toolkit* oder *AMD APP SDK*). |
| Build-Werkzeuge | `cmake` ist nicht erforderlich – einfache Compileraufrufe genügen. |

> **Hinweis:** Unter Windows benötigen Sie zusätzlich das passende „x64 Native Tools Command Prompt“ des verwendeten Compilers.

### 1.1. Ordnerstruktur

Speichern Sie nach dem Kompilieren die Shared Library (`halo_fastpath.dll`, `halo_driver.dll`, `libhalo_fastpath.so`, …) im **selben Ordner** wie `halo.py`. Die Python-Bridge erwartet sie dort.

### 1.2. Optionale Umgebungsvariablen

| Variable | Zweck |
| :-- | :-- |
| `OPENCL_SDK_DIR` | Pfad zu Headern/Libraries, wenn der Compiler sie nicht automatisch findet. |
| `CL_CONFIG_USE_V0_CLBLAS` | Für manche OpenCL-Implementierungen nötig, wenn `clBLAS` verwendet wird. HALO benötigt dies nicht, kann aber bei kundenspezifischen Builds relevant sein. |

---

## 2. CPU Fast-Path kompilieren (`fastpath.cpp`)

### 2.1. Linux / macOS (GCC oder Clang)

```bash
g++ -std=c++17 -O3 -march=native -fPIC -pthread \
    -shared fastpath.cpp -o libhalo_fastpath.so
```

**Parameter-Erklärung:**
- `-march=native` aktiviert AVX2/FMA auf der lokalen Maschine. Für portable Builds z. B. `-mavx2 -mfma` verwenden.
- `-fPIC` ist für Shared Libraries unter Unix erforderlich.
- `-pthread` bindet die Thread-Pool-Implementierung korrekt ein.

### 2.2. Windows (MinGW-w64)

```bash
g++ -std=c++17 -O3 -march=native -shared -static-libstdc++ -static-libgcc \
    fastpath.cpp -o halo_fastpath.dll
```

### 2.3. Windows (MSVC)

```cmd
cl /LD /O2 /std:c++17 /arch:AVX2 fastpath.cpp /Fe:halo_fastpath.dll
```

> **Tipp:** Fügen Sie `/openmp` hinzu, falls Sie eigene Erweiterungen mit OpenMP planen.

---

## 3. Kombinierten GPU-Treiber kompilieren (`halo_driver.cpp`)

Der GPU-Treiber inkludiert sowohl `fastpath.cpp` als auch `CipherCore_OpenCl.c`. Stellen Sie sicher, dass diese Dateien im selben Ordner liegen.

### 3.1. Zusätzliche Abhängigkeiten

1. **OpenCL Header** (`CL/cl.h`, `CL/cl_ext.h`).
2. **OpenCL Library** (`-lOpenCL` unter Unix, `OpenCL.lib` unter Windows).
3. Optional: Setzen Sie `HALO_ENABLE_GPU=1` als Vorab-Check in Ihren Build-Skripten, um GPU-spezifische CI-Läufe zu kennzeichnen.

### 3.2. Linux / macOS (GCC oder Clang)

```bash
g++ -std=c++17 -O3 -march=native -fPIC -pthread \
    halo_driver.cpp -shared -o libhalo_driver.so \
    -lOpenCL
```

**Varianten:**
- Verwenden Sie `-L/path/to/opencl/lib -I/path/to/opencl/include`, wenn der Compiler die OpenCL-Dateien nicht automatisch findet.
- Für Clang ersetzen Sie den Compileraufruf entsprechend (`clang++`).

### 3.3. Windows (MinGW-w64)

```bash
g++ -std=c++17 -O3 -march=native -shared \
    halo_driver.cpp -o halo_driver.dll \
    -lOpenCL -static-libstdc++ -static-libgcc
```

Falls `OpenCL.lib` statt `libOpenCL.a` vorliegt, geben Sie den Pfad explizit an:

```bash
g++ ... -L"C:/Program Files/OpenCL/lib" -lOpenCL
```

### 3.4. Windows (MSVC)

```cmd
cl /LD /O2 /std:c++17 /arch:AVX2 halo_driver.cpp \
    /I"%OPENCL_SDK_DIR%/include" \
    /link /OUT:halo_driver.dll /LIBPATH:"%OPENCL_SDK_DIR%/lib" OpenCL.lib
```

> **Wichtig:** Starten Sie vorher `vcvars64.bat` oder das „x64 Native Tools Command Prompt“, damit `cl.exe` und der Linker verfügbar sind.

---

## 4. Build-Tests & Validierung

1. **Library prüfen:**
   ```bash
   nm -D libhalo_driver.so | grep halo_init
   ```
   Unter Windows: `dumpbin /EXPORTS halo_driver.dll | findstr halo_init`.

2. **Python-Integration testen:**
   ```bash
   python - <<'PY'
   import halo
   halo_gpu = halo.HALO(enable_gpu=True)
   print("GPU aktiv:", halo_gpu.is_gpu_available())
   PY
   ```

3. **Demo starten:**
   ```bash
   python halo_demo_app.py
   ```

---

## 5. Fehlersuche

| Problem | Ursache | Lösung |
| :-- | :-- | :-- |
| `fatal error: CL/cl.h: No such file or directory` | OpenCL-Header nicht im Include-Pfad. | `-I`/`/I` verwenden oder `OPENCL_SDK_DIR` setzen. |
| `undefined reference to \\"clCreateProgramWithSource\\"` | OpenCL-Library fehlt. | `-lOpenCL` (Unix) oder `OpenCL.lib` (Windows) hinzufügen. |
| DLL lässt sich nicht laden | Fehlende Runtime-Abhängigkeiten (OpenCL ICD, VC++ Redistributable). | Entsprechende Runtime installieren. |
| GPU wird nicht erkannt | Kein kompatibles Gerät oder ICD installiert. | OpenCL-Treiber aktualisieren, Gerät prüfen (`clinfo`). |

---

## 6. Automatisierte Builds (optional)

Für CI/CD können Sie einfache Skripte erstellen, z. B. `build_gpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
CXX=${CXX:-g++}
OPENCL_ROOT=${OPENCL_ROOT:-/opt/OpenCL}
$CXX -std=c++17 -O3 -fPIC -pthread -I"$OPENCL_ROOT/include" \
    halo_driver.cpp -shared -o libhalo_driver.so \
    -L"$OPENCL_ROOT/lib" -lOpenCL
```

Unter Windows bietet sich ein `build_gpu.bat` an:

```cmd
@echo off
setlocal
del /Q halo_driver.dll 2>nul
cl /LD /O2 /std:c++17 /arch:AVX2 halo_driver.cpp ^
    /I"%OPENCL_SDK_DIR%/include" ^
    /link /OUT:halo_driver.dll /LIBPATH:"%OPENCL_SDK_DIR%/lib" OpenCL.lib
endlocal
```

Damit steht einer reproduzierbaren Build-Pipeline nichts im Wege.

