"""Mapa de ocupacion 2D del MiR100 (log-odds), portado de DistributedAI/Proyecto/mir100/mapping.py.

Dos capas en el mismo marco del mundo:
- LiDAR: cada rayo marca libre el espacio que atraviesa y ocupada la celda donde
  impacta. Un rayo sin eco limpia hasta MAX_RANGO_LIBRE.
- Alto (camara de profundidad): solo impactos. Va aparte porque los rayos del LiDAR
  pasan por DEBAJO de una mesa y, en la misma capa, borrarian la tabla del mapa.

Ocupado = ocupado en cualquiera de las dos. Las celdas desconocidas se planifican
como libres (optimista): el robot va hacia lo que no conoce y replanifica al verlo.

Pose = (x, y, yaw) en el mundo (Z arriba), yaw en rad. Sin dependencias de Webots.
"""
import math

import numpy as np

import lidar

X_MIN, Y_MIN = -10.0, -7.0        # arena de 19 x 13 m centrada en el origen, con margen
ANCHO, ALTO = 20.0, 14.0
RESOLUCION = 0.1                  # m por celda
MAX_RANGO_LIBRE = 6.0             # hasta donde un rayo limpia espacio libre
MAX_RANGO_IMPACTO = 4.0           # impactos mas lejanos no marcan ocupado: de lejos los rayos son
                                  # escasos y sus puntos sueltos, inflados, cerraban pasillos que existen

LOG_ODDS_IMPACTO = 0.85
LOG_ODDS_LIBRE = -0.4
LOG_ODDS_MIN, LOG_ODDS_MAX = -4.0, 4.0
UMBRAL_OCUPADO = 0.6
LOG_ODDS_ALTO = 0.35              # la profundidad da muchos puntos por celda: sube de a poco
UMBRAL_ALTO = 0.6                 # ~2 lecturas para marcar una celda alta


def desplazamientos_disco(radio_celdas):
    return [(df, dc) for df in range(-radio_celdas, radio_celdas + 1)
            for dc in range(-radio_celdas, radio_celdas + 1) if df * df + dc * dc <= radio_celdas * radio_celdas]


def dilatar(mascara, radio_celdas):
    """Engorda las celdas True con un disco (sin scipy)."""
    if radio_celdas <= 0:
        return mascara.copy()
    filas, columnas = mascara.shape
    resultado = np.zeros_like(mascara)
    for df, dc in desplazamientos_disco(radio_celdas):
        origen_f = slice(max(0, -df), min(filas, filas - df))
        destino_f = slice(max(0, df), min(filas, filas + df))
        origen_c = slice(max(0, -dc), min(columnas, columnas - dc))
        destino_c = slice(max(0, dc), min(columnas, columnas + dc))
        resultado[destino_f, destino_c] |= mascara[origen_f, origen_c]
    return resultado


def robot_a_mundo(pose, x, y):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return px + c * x - s * y, py + s * x + c * y


class Mapa:
    def __init__(self, x_min=X_MIN, y_min=Y_MIN, ancho=ANCHO, alto=ALTO, resolucion=RESOLUCION):
        self.x_min, self.y_min, self.resolucion = x_min, y_min, resolucion
        self.columnas = int(round(ancho / resolucion))
        self.filas = int(round(alto / resolucion))
        self.log_odds = np.zeros((self.filas, self.columnas), dtype=np.float32)
        self.alto = np.zeros((self.filas, self.columnas), dtype=np.float32)

    def celda(self, x, y):
        return int(math.floor((y - self.y_min) / self.resolucion)), int(math.floor((x - self.x_min) / self.resolucion))

    def centro(self, fila, columna):
        return self.x_min + (columna + 0.5) * self.resolucion, self.y_min + (fila + 0.5) * self.resolucion

    def dentro(self, fila, columna):
        return 0 <= fila < self.filas and 0 <= columna < self.columnas

    def ocupado(self):
        return (self.log_odds > UMBRAL_OCUPADO) | (self.alto > UMBRAL_ALTO)

    def conocido(self):
        return (self.log_odds != 0.0) | (self.alto > 0.0)

    def _celdas_unicas(self, xs, ys):
        columnas = np.floor((xs - self.x_min) / self.resolucion).astype(np.int64)
        filas = np.floor((ys - self.y_min) / self.resolucion).astype(np.int64)
        validas = (filas >= 0) & (filas < self.filas) & (columnas >= 0) & (columnas < self.columnas)
        return np.unique(filas[validas] * self.columnas + columnas[validas])

    def actualizar_lidar(self, pose, rangos, primer_angulo, posicion_sensor, rango_maximo, fov_grados=lidar.FOV_GRADOS):
        r = np.asarray(rangos, dtype=np.float64)
        total = len(r)
        if total < 2:
            return
        angulos = np.radians(primer_angulo + np.arange(total) * fov_grados / (total - 1))
        sin_eco = np.isposinf(r)
        usar = sin_eco | (np.isfinite(r) & (r > lidar.RANGO_MINIMO_VALIDO))
        r, angulos, sin_eco = r[usar], angulos[usar], sin_eco[usar]
        eco = ~sin_eco & (r < rango_maximo * 0.999)
        largo_libre = np.minimum(np.where(eco, r, MAX_RANGO_LIBRE), MAX_RANGO_LIBRE)
        impacto = eco & (r <= MAX_RANGO_IMPACTO)

        ox, oy = robot_a_mundo(pose, *posicion_sensor)
        direccion = pose[2] + angulos
        dx, dy = np.cos(direccion), np.sin(direccion)

        distancias = np.arange(0.0, MAX_RANGO_LIBRE, self.resolucion * 0.5)
        mascara = distancias[None, :] < (largo_libre[:, None] - self.resolucion)
        libres = self._celdas_unicas((ox + dx[:, None] * distancias[None, :])[mascara],
                                     (oy + dy[:, None] * distancias[None, :])[mascara])
        impactos = self._celdas_unicas(ox + dx[impacto] * r[impacto], oy + dy[impacto] * r[impacto])
        libres = np.setdiff1d(libres, impactos, assume_unique=True)

        plano = self.log_odds.reshape(-1)
        plano[libres] += LOG_ODDS_LIBRE
        plano[impactos] += LOG_ODDS_IMPACTO
        np.clip(self.log_odds, LOG_ODDS_MIN, LOG_ODDS_MAX, out=self.log_odds)

    def actualizar_alto(self, pose, xs_robot, ys_robot):
        """Puntos de obstaculo de la camara de profundidad, en el marco del robot."""
        if len(xs_robot) == 0:
            return
        c, s = math.cos(pose[2]), math.sin(pose[2])
        xs = pose[0] + c * np.asarray(xs_robot) - s * np.asarray(ys_robot)
        ys = pose[1] + s * np.asarray(xs_robot) + c * np.asarray(ys_robot)
        plano = self.alto.reshape(-1)
        plano[self._celdas_unicas(xs, ys)] += LOG_ODDS_ALTO
        np.clip(self.alto, 0.0, LOG_ODDS_MAX, out=self.alto)
