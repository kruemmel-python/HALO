#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gesichtsanimation via expliziten Koordinatenfeldern (map_x/map_y) und HALO warp_custom.
- Python 3.12
- Nutzt moderne Sprachfeatures (PEP 634 Pattern Matching, PEP 604 Union '|' usw.)
- Minimale Abhängigkeiten: numpy, imageio[ffmpeg] oder opencv-python (wählbar)
- Keine ML-Landmarks: einfache, parametrisierte Ellipsen für Augen und Mund
- Sauberes Fehlermanagement mit klaren Messages (PEP 626 Geist)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Callable, Iterable, Tuple, Optional
import math
import numpy as np

# HALO High-Level Warp (reines NumPy, aber Teil deiner HALO-Erweiterungen)
# -> erlaubt explizite Koordinatenfelder (map_x, map_y) mit bilinearer Abtastung
from halo_extensions import warp_custom  # :contentReference[oaicite:3]{index=3}

# --------------------------
# Typen & Utilities
# --------------------------

EaseName = Literal["linear", "smoothstep", "smootherstep", "inout_sine", "in_cubic", "out_cubic"]

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def ease_f(name: EaseName) -> Callable[[float], float]:
    """
    Wähle eine Easing-Funktion per Pattern Matching (PEP 634).
    Warum? Klarer, expliziter, erweiterbar und leicht testbar.
    """
    match name:
        case "linear":
            return lambda t: _clamp01(t)
        case "smoothstep":
            return lambda t: (t:=_clamp01(t)) * t * (3 - 2*t)
        case "smootherstep":
            return lambda t: (t:=_clamp01(t))**3 * (t*(6*t - 15) + 10)
        case "inout_sine":
            return lambda t: 0.5 - 0.5*math.cos(math.pi*_clamp01(t))
        case "in_cubic":
            return lambda t: (_clamp01(t))**3
        case "out_cubic":
            return lambda t: 1 - (1 - _clamp01(t))**3
    # Fallback (Zen: explicit is better… aber pragmatisch)
    return lambda t: _clamp01(t)

@dataclass(slots=True)
class Ellipse:
    """
    Ellipse-Region als einfache Parametrisierung.
    (cx, cy): Zentrum in Pixeln
    (rx, ry): Radien in Pixeln (>= 1)
    angle: Rotation in Radiant (mathematisch positiv = gegen Uhrzeigersinn)
    """
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float = 0.0

    def mask_and_local(self, grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rechne lokale Koordinaten u,v in Ellipsen-Achsen und eine Maske:
        u=0,v=0 am Zentrum; normierte Ellipse erfüllt (u/rx)^2+(v/ry)^2 <= 1.
        """
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        dx = grid_x - self.cx
        dy = grid_y - self.cy
        # Rotation ins lokale Achsensystem
        u =  cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        inside = (u*u)/(self.rx*self.rx + 1e-12) + (v*v)/(self.ry*self.ry + 1e-12) <= 1.0
        return inside, u, v

# --------------------------
# Deformations
# --------------------------

def deform_eye_blink(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    eye: Ellipse,
    openness: float,
    lid_strength: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simuliere ein Augenlid, das schließt (openness -> 0) und öffnet (-> 1).
    Idee:
      - Punkte innerhalb der Ellipse werden in y-Richtung zum Zentrum gezogen,
        Stärke ~ (1 - openness), skaliert parabolisch entlang der lokalen v-Achse.
    Warum so? Einfach, glatt, vermeidet Artefakte bei kleinen Radien.
    """
    openness = _clamp01(openness)
    inside, u, v = eye.mask_and_local(grid_x, grid_y)
    # Normierung [-1..1] entlang v
    v_norm = v / (eye.ry + 1e-12)
    # Parabolische Gewichtung (oben/unten stärker)
    w = (1.0 - (v_norm * v_norm))  # 0 an Rand, 1 in der Mitte
    w = np.clip(w, 0.0, 1.0)
    # Verschiebung Richtung Zentrum (v->0), skaliert über lid_strength * (1 - openness)
    dv = -v * (1.0 - openness) * lid_strength * w
    # Zurück in Welt-Koordinaten: nur y-Komponente ändert sich, x nahezu stabil
    cos_a = math.cos(eye.angle)
    sin_a = math.sin(eye.angle)
    # (du,dv) -> (dx,dy)
    dx = -sin_a * dv  # u bleibt ~gleich, daher nur Rotationsanteil aus dv
    dy =  cos_a * dv
    # Nur innerhalb der Ellipse anwenden
    map_x = grid_x.copy()
    map_y = grid_y.copy()
    map_x[inside] = grid_x[inside] + dx[inside]
    map_y[inside] = grid_y[inside] + dy[inside]
    return map_x, map_y

def deform_mouth_open(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mouth: Ellipse,
    open_amount: float,
    vertical_bias: float = 0.9,
    roundness: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mundöffnung: Dehne die Ellipse in vertikaler Richtung um eine weiche Kurve,
    mit leichtem „Rundungs“-Term (macht die Öffnung ovaler statt nur linear).
    """
    a = max(0.0, float(open_amount))
    inside, u, v = mouth.mask_and_local(grid_x, grid_y)
    v_norm = v / (mouth.ry + 1e-12)   # [-1..1]
    u_norm = u / (mouth.rx + 1e-12)
    # Weiche Gewichtung: in der Mitte (|u| klein) stärker öffnen
    lateral = 1.0 - np.clip(u_norm*u_norm, 0.0, 1.0)
    # Vertikaler Stretch (oben/unten): skaliert v zur Öffnung
    stretch = (a * vertical_bias) * (1.0 - np.abs(v_norm))  # offene Mitte
    # Rundung: kleine „Bauchigkeit“ in der Mitte
    bulge = (a * (1.0 - vertical_bias)) * (1.0 - (u_norm*u_norm + v_norm*v_norm))**roundness
    dv = (stretch + bulge) * np.sign(v) * mouth.ry * 0.5  # skaliere auf Pixel
    # Zurückrotieren
    cos_a = math.cos(mouth.angle)
    sin_a = math.sin(mouth.angle)
    dx = -sin_a * dv
    dy =  cos_a * dv
    # Nur im Mund anwenden
    map_x = grid_x.copy()
    map_y = grid_y.copy()
    map_x[inside] = grid_x[inside] + dx[inside]
    map_y[inside] = grid_y[inside] + dy[inside]
    return map_x, map_y

# --------------------------
# Animator
# --------------------------

@dataclass(slots=True)
class FaceRig:
    """
    Minimal-Rig per Ellipsen (kein ML).
    Typische Startwerte sind relative Positionen, die du pro Bild skalierst.
    """
    left_eye: Ellipse
    right_eye: Ellipse
    mouth: Ellipse

@dataclass(slots=True)
class BlinkSpec:
    curve: EaseName = "smootherstep"   # natürliche Lidbewegung
    strength: float = 0.65            # wie stark das Lid ins Zentrum zieht

@dataclass(slots=True)
class SpeakSpec:
    curve: EaseName = "inout_sine"    # Öffnungs/Schließzyklus je Frame
    amplitude: float = 0.9            # maximale Öffnung
    viseme_factor: float = 0.0        # 0..1 – optionale Zusatzmodulation

class FaceAnimator:
    """
    Baut pro Frame map_x/map_y und rendert mit HALO warp_custom (bilinear).
    """
    def __init__(self, img: np.ndarray, rig: FaceRig) -> None:
        if img.ndim not in (2,3):
            raise ValueError("Erwarte Grau- oder RGB-Bild als NumPy-Array (H,W[,C]).")
        if img.dtype != np.uint8:
            raise TypeError("Erwarte uint8-Eingangsbild (einfach zu demonstrieren).")
        self.img = img
        self.rig = rig
        self.h, self.w = img.shape[:2]
        # Identitätsgitter (Zielkoordinate -> Quellkoordinate identisch)
        gx, gy = np.meshgrid(np.arange(self.w, dtype=np.float32),
                             np.arange(self.h, dtype=np.float32))
        self.base_x = gx
        self.base_y = gy

    def _compose_maps(self, maps: Iterable[Tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Komponiere mehrere lokale Deformationen additiv auf das Basisgitter.
        Warum additiv? Für kleine Offsets ist das stabil, schnell und gut beherrschbar.
        """
        mx = self.base_x.copy()
        my = self.base_y.copy()
        for map_x, map_y in maps:
            mx += (map_x - self.base_x)
            my += (map_y - self.base_y)
        return mx, my

    def render_frame(
        self,
        eye_openness: float,      # 1.0 = offen, 0.0 = geschlossen
        mouth_open: float,        # 0..1 Öffnungsgrad
        blink: BlinkSpec = BlinkSpec(),
        speak: SpeakSpec = SpeakSpec(),
    ) -> np.ndarray:
        """
        Erzeuge ein einzelnes Animationsframe.
        """
        # Easing anwenden (glättet Bewegungen sichtbar)
        e_eye  = ease_f(blink.curve)(eye_openness)
        e_mouth= ease_f(speak.curve)(mouth_open)

        # Lokale Deformationen erzeugen
        lmx, lmy = deform_eye_blink(self.base_x, self.base_y, self.rig.left_eye,  e_eye,  blink.strength)
        rmx, rmy = deform_eye_blink(self.base_x, self.base_y, self.rig.right_eye, e_eye,  blink.strength)
        mmx, mmy = deform_mouth_open(self.base_x, self.base_y, self.rig.mouth, e_mouth, vertical_bias=0.9, roundness=0.8)

        # Komposition
        map_x, map_y = self._compose_maps([(lmx,lmy),(rmx,rmy),(mmx,mmy)])

        # Rendering via HALO High-Level Warp (bilinear)
        out = warp_custom(self.img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation="bilinear", cval=0.0)  # :contentReference[oaicite:4]{index=4}
        return out

# --------------------------
# High-level Helfer
# --------------------------

def build_default_rig(h: int, w: int) -> FaceRig:
    """
    Baue einen pragmatischen Default-Rig aus relativen Ellipsen:
    - Augen oberes Drittel, Mund im unteren Drittel
    - Radien relativ zur Bildbreite/-höhe
    """
    cx = w * 0.3; cy = h * 0.35
    rx = w * 0.08; ry = h * 0.05
    left = Ellipse(cx=cx, cy=cy, rx=rx, ry=ry, angle=0.0)

    cx2 = w * 0.7; cy2 = h * 0.35
    right = Ellipse(cx=cx2, cy=cy2, rx=rx, ry=ry, angle=0.0)

    mx = w * 0.5; my = h * 0.70
    mr = w * 0.14; myr = h * 0.07
    mouth = Ellipse(cx=mx, cy=my, rx=mr, ry=myr, angle=0.0)

    return FaceRig(left_eye=left, right_eye=right, mouth=mouth)

def make_blink_curve(num_frames: int, hold_open: int = 10, hold_closed: int = 4) -> np.ndarray:
    """
    Erzeuge eine plausible Blinkkurve: offen -> schließen -> offen, mit Haltephasen.
    """
    close_frames = max(2, num_frames // 6)
    open_frames  = close_frames
    # Offen-Phase | Schließen | Geschlossen | Öffnen
    segs = [
        np.ones(hold_open, dtype=np.float32),
        np.linspace(1.0, 0.0, close_frames, dtype=np.float32),
        np.zeros(hold_closed, dtype=np.float32),
        np.linspace(0.0, 1.0, open_frames, dtype=np.float32),
    ]
    curve = np.concatenate(segs)
    if curve.size < num_frames:
        curve = np.pad(curve, (0, num_frames-curve.size), constant_values=1.0)
    return curve[:num_frames]

def make_speech_curve(num_frames: int, syllables: int = 6) -> np.ndarray:
    """
    Sehr simple „Sprech“-Kurve (Öffnungs-Amplitude), sinusförmig über Silben.
    Für echte Visemen würdest du hier pro Phonem modulieren.
    """
    t = np.linspace(0, 2*math.pi*syllables, num_frames, dtype=np.float32)
    raw = 0.5*(1.0 + np.sin(t))  # 0..1
    return raw.astype(np.float32)

# --------------------------
# Demo / CLI
# --------------------------

def animate_image(
    img: np.ndarray,
    n_frames: int = 48,
    fps: int = 24,
    out_path: Optional[str] = None,
    use_cv2: bool = False,
) -> list[np.ndarray]:
    """
    Voller Animationslauf auf Basis von Blinzeln + Sprechen.
    - out_path endet auf .mp4 oder .gif -> wird gespeichert (optional).
    """
    h, w = img.shape[:2]
    rig = build_default_rig(h, w)
    animator = FaceAnimator(img, rig)

    blink_curve = make_blink_curve(n_frames, hold_open=8, hold_closed=3)
    speech_curve = make_speech_curve(n_frames, syllables=4)

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        frame = animator.render_frame(
            eye_openness=float(blink_curve[i]),
            mouth_open=float(speech_curve[i]),
        )
        frames.append(frame)

    if out_path:
        if use_cv2:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            for f in frames:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                vw.write(bgr)
            vw.release()
        else:
            import imageio.v2 as iio
            if out_path.lower().endswith(".gif"):
                iio.mimsave(out_path, frames, duration=1.0/fps, loop=0)
            else:
                iio.mimsave(out_path, frames, fps=fps)

    return frames

if __name__ == "__main__":
    # Kleine CLI-Demo: Bild „face.png“ im aktuellen Ordner laden und animieren.
    import sys, os
    if len(sys.argv) < 2:
        print("Verwendung: python face_animator_halo.py <bilddatei> [out.mp4|out.gif]")
        sys.exit(0)
    import imageio.v2 as iio
    src_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    img = iio.imread(src_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    frames = animate_image(img, n_frames=64, fps=24, out_path=out_path)
    print(f"Fertig. Frames: {len(frames)} -> {out_path or '(nur Speicher)'}")
