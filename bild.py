from halo import VectorCanvas, VectorShape

# Canvas anlegen
canvas = VectorCanvas(400, 200, background=0.0)

# Linie, Kreis, Text zeichnen
canvas.stroke_segment((10, 10), (390, 10), color=1.0, width=2.0)
canvas.fill_polygon([(200,100),(250,150),(150,150)], color=0.8)
canvas.draw_text(120, 180, "HALO v0.5b", size=16, color=1.0)

# Speichern als Graustufenbild
import numpy as np, imageio
img = np.frombuffer(canvas.buffer, dtype=np.float32).reshape(canvas.height, canvas.width)
imageio.imwrite("out.png", (np.clip(img,0,1)*255).astype(np.uint8))
