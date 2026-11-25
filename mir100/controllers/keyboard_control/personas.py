"""Ubicar personas detectadas por YOLO en el marco del robot.

El recuadro de YOLO da la DIRECCION (columna del centro en la imagen de la camara).
La DISTANCIA sale, en este orden, de los puntos de la camara de profundidad (hasta
1.95 m) o del LiDAR (las piernas cortan su plano) que caen dentro del ancho angular
del recuadro; si no hay, se estima por la altura del recuadro con una persona de
ALTURA_PERSONA metros (camara pinhole), acotada al alcance de la camara. Esa ultima
estimacion es aproximada: un recuadro cortado o una persona sentada la exageran.

Con eso el controlador decide: frenar solo si la persona esta delante y cerca,
ir despacio si esta cerca, y el planificador la rodea. Sin dependencias de Webots.
"""
import math

import numpy as np

import lidar

POSICION_CAMARA_X = 0.4376   # m: bodySlot (0.037646) + JetBotRaspberryPiCamera (0.4), mirando al frente
ALTURA_PERSONA = 1.7         # m, para estimar la distancia por el tamano del recuadro
ALCANCE_CAMARA = 10.0        # m (far de la camara en el mundo)


def _distancia_en_recuadro(puntos_xy, a_izq, a_der):
    if puntos_xy is None or not len(puntos_xy[0]):
        return None
    xs = np.asarray(puntos_xy[0], dtype=float) - POSICION_CAMARA_X
    ys = np.asarray(puntos_xy[1], dtype=float)
    angulos = np.arctan2(ys, xs)
    dentro = (xs > 0.05) & (angulos <= a_izq) & (angulos >= a_der)
    if np.count_nonzero(dentro) < 3:
        return None
    return float(np.percentile(np.hypot(xs[dentro], ys[dentro]), 20))


def ubicar(caja, ancho, alto, fov, puntos_xy=None, puntos_lidar=None):
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

    distancia, fuente = _distancia_en_recuadro(puntos_xy, a_izq, a_der), "profundidad"
    if distancia is None:
        distancia, fuente = _distancia_en_recuadro(puntos_lidar, a_izq, a_der), "lidar"
    if distancia is None:
        distancia = min(ALCANCE_CAMARA, focal * ALTURA_PERSONA / max(1.0, y2 - y1))
        fuente = "tamano_aproximado"

    px = POSICION_CAMARA_X + distancia * math.cos(angulo)
    py = distancia * math.sin(angulo)
    holgura = math.hypot(max(abs(px) - lidar.MEDIO_LARGO, 0.0), max(abs(py) - lidar.MEDIO_ANCHO, 0.0))
    return {"x": px, "y": py, "distancia": distancia, "holgura": holgura,
            "angulo": math.degrees(angulo), "fuente": fuente}
