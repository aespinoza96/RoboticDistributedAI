"""Ubicar personas detectadas por YOLO en el marco del robot.

El recuadro de YOLO da la DIRECCION (columna del centro en la imagen de la camara).
La DISTANCIA sale de los puntos de la camara de profundidad que caen dentro del
ancho angular del recuadro; si no hay (la persona esta a mas de 1.95 m, fuera del
alcance de la profundidad), se estima por la altura del recuadro con una persona de
ALTURA_PERSONA metros (camara pinhole).

Con eso el controlador decide: frenar solo si la persona esta delante y cerca,
ir despacio si esta cerca, y el planificador la rodea. Sin dependencias de Webots.
"""
import math

import numpy as np

import lidar

POSICION_CAMARA_X = 0.4376   # m: bodySlot (0.037646) + JetBotRaspberryPiCamera (0.4), mirando al frente
ALTURA_PERSONA = 1.7         # m, para estimar la distancia por el tamano del recuadro


def ubicar(caja, ancho, alto, fov, puntos_xy=None):
    """caja = (x1, y1, x2, y2) en pixeles. puntos_xy = (xs, ys) de obstaculos de la
    profundidad en el marco del robot, o None. Devuelve un dict con x, y (marco del
    robot), distancia (desde la camara), holgura (al borde del robot), angulo
    (grados, + = izquierda) y fuente ("profundidad" o "tamano")."""
    x1, y1, x2, y2 = [float(v) for v in caja]
    focal = (ancho / 2.0) / math.tan(fov / 2.0)
    centro = (ancho - 1) / 2.0
    angulo = math.atan2(-((x1 + x2) / 2.0 - centro), focal)
    a_izq = math.atan2(-(x1 - centro), focal)
    a_der = math.atan2(-(x2 - centro), focal)

    distancia, fuente = None, "tamano"
    if puntos_xy is not None and len(puntos_xy[0]):
        xs = np.asarray(puntos_xy[0], dtype=float) - POSICION_CAMARA_X
        ys = np.asarray(puntos_xy[1], dtype=float)
        delante = xs > 0.05
        angulos = np.arctan2(ys, xs)
        dentro = delante & (angulos <= a_izq) & (angulos >= a_der)
        if np.count_nonzero(dentro) >= 3:
            distancia = float(np.percentile(np.hypot(xs[dentro], ys[dentro]), 20))
            fuente = "profundidad"
    if distancia is None:
        distancia = focal * ALTURA_PERSONA / max(1.0, y2 - y1)

    px = POSICION_CAMARA_X + distancia * math.cos(angulo)
    py = distancia * math.sin(angulo)
    holgura = math.hypot(max(abs(px) - lidar.MEDIO_LARGO, 0.0), max(abs(py) - lidar.MEDIO_ANCHO, 0.0))
    return {"x": px, "y": py, "distancia": distancia, "holgura": holgura,
            "angulo": math.degrees(angulo), "fuente": fuente}
