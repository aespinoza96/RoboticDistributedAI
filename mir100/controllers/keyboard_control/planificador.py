"""Ruta planificada sobre el mapa: A* + seguimiento pure pursuit.

Portado de DistributedAI/Proyecto/mir100 (planner.py, follower.py, navigator.py).
El robot se trata como un punto en su centro; los obstaculos se engordan con
INFLADO (medio ancho del robot + margen) y una zona blanda mas amplia encarece las
celdas cerca de obstaculos, para que la ruta vaya por el medio de los pasillos.

Sin dependencias de Webots: se prueba con tests/test_planificador.py.
"""
import heapq
import math
from collections import deque

import numpy as np

import mapa as mapa_mod

INFLADO = 0.55                 # m, centro del robot a obstaculo (medio ancho 0.29 + 0.26)
INFLADO_MINIMO = 0.38          # m: si no hay ruta con INFLADO, se reintenta con esto (pasillos angostos)
INFLADO_BLANDO = 0.95          # m, zona con costo extra
COSTO_BLANDO = 4.0
RADIO_AJUSTE_META = 1.5        # m: si la meta cae en un obstaculo, la celda libre mas cercana
REPLANIFICAR_CADA_S = 2.0

ADELANTO = 0.6                 # pure pursuit: punto de la ruta a esta distancia
TOLERANCIA_META = 0.30
ANGULO_GIRO_EN_SITIO = 0.6     # rad: mas desviado que esto, girar sin avanzar
GANANCIA_RUMBO = 1.5
RADIO_FRENADO = 1.0
FACTOR_MINIMO = 0.25

VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

LLEGO, SIN_RUTA = "llego", "sin_ruta"


def octil(a, b):
    df, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(df, dc) + (math.sqrt(2) - 1) * min(df, dc)


def astar(bloqueado, inicio, meta, costo_extra=None):
    """Lista de celdas de inicio a meta, o None si no hay ruta."""
    filas, columnas = bloqueado.shape
    b = bloqueado.tolist()
    costo = costo_extra.tolist() if costo_extra is not None else None
    if b[meta[0]][meta[1]]:
        return None
    g = {inicio: 0.0}
    viene_de = {}
    abiertos = [(octil(inicio, meta), 0.0, inicio)]
    cerrados = set()
    while abiertos:
        _, g_actual, actual = heapq.heappop(abiertos)
        if actual in cerrados:
            continue
        if actual == meta:
            ruta = [actual]
            while actual in viene_de:
                actual = viene_de[actual]
                ruta.append(actual)
            return ruta[::-1]
        cerrados.add(actual)
        f, c = actual
        for df, dc, paso in VECINOS:
            nf, nc = f + df, c + dc
            if nf < 0 or nf >= filas or nc < 0 or nc >= columnas or b[nf][nc]:
                continue
            if df and dc and (b[f][nc] or b[nf][c]):   # no cortar esquinas en diagonal
                continue
            nuevo = g_actual + paso + (costo[nf][nc] if costo is not None else 0.0)
            if nuevo < g.get((nf, nc), float("inf")):
                g[(nf, nc)] = nuevo
                viene_de[(nf, nc)] = actual
                heapq.heappush(abiertos, (nuevo + octil((nf, nc), meta), nuevo, (nf, nc)))
    return None


def libre_mas_cercana(bloqueado, celda, radio_celdas):
    if not bloqueado[celda]:
        return celda
    filas, columnas = bloqueado.shape
    vistos = {celda}
    cola = deque([celda])
    while cola:
        f, c = cola.popleft()
        for df, dc, _ in VECINOS:
            vecino = (f + df, c + dc)
            if not (0 <= vecino[0] < filas and 0 <= vecino[1] < columnas) or vecino in vistos:
                continue
            if max(abs(vecino[0] - celda[0]), abs(vecino[1] - celda[1])) > radio_celdas:
                continue
            if not bloqueado[vecino]:
                return vecino
            vistos.add(vecino)
            cola.append(vecino)
    return None


def angulo_normal(a):
    return math.atan2(math.sin(a), math.cos(a))


class Navegador:
    """Planifica con A*, sigue la ruta y replanifica cada REPLANIFICAR_CADA_S o si la
    ruta restante queda ocupada. paso() devuelve (v, omega, resultado)."""

    def __init__(self, mapa, velocidad_max, omega_max):
        self.mapa = mapa
        self.velocidad_max = velocidad_max
        self.omega_max = omega_max
        self.inflado = int(math.ceil(INFLADO / mapa.resolucion))
        self.inflado_minimo = int(math.ceil(INFLADO_MINIMO / mapa.resolucion))
        self.inflado_blando = int(math.ceil(INFLADO_BLANDO / mapa.resolucion))
        self.ajuste_meta = int(math.ceil(RADIO_AJUSTE_META / mapa.resolucion))
        self.activo = False
        self.meta = None
        self.ruta_celdas = None
        self.ruta = []
        self.progreso = 0
        self.ultima_planificacion = -1e9
        self.replanificaciones = 0
        self.zonas = []                # (x, y, radio): areas a evitar, p. ej. personas

    def fijar_zonas(self, zonas, umbral=0.5):
        """Zonas a evitar (x, y, radio). Replanifica si aparece una nueva o alguna se movio."""
        def lejos(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1]) > umbral
        cambio = len(zonas) != len(self.zonas) or any(all(lejos(z, v) for v in self.zonas) for z in zonas)
        self.zonas = list(zonas)
        if cambio and self.activo:
            self.replanificar_ya()

    def iniciar(self, meta):
        self.activo, self.meta = True, (float(meta[0]), float(meta[1]))
        self.ruta_celdas, self.ruta, self.progreso = None, [], 0
        self.replanificaciones = 0

    def detener(self):
        self.activo, self.ruta_celdas, self.ruta = False, None, []

    def replanificar_ya(self):
        self.ruta_celdas = None

    def planificar(self, pose):
        """Ruta con margen normal; si no hay, con margen minimo (pasillos angostos)."""
        ocupado = self.mapa.ocupado()
        inicio = self.mapa.celda(pose[0], pose[1])
        meta_celda = self.mapa.celda(*self.meta)
        if not self.mapa.dentro(*inicio) or not self.mapa.dentro(*meta_celda):
            return None
        for inflado in (self.inflado, self.inflado_minimo):
            bloqueado = mapa_mod.dilatar(ocupado, inflado)
            for zx, zy, radio in self.zonas:
                zf, zc = self.mapa.celda(zx, zy)
                for df, dc in mapa_mod.desplazamientos_disco(int(math.ceil(radio / self.mapa.resolucion))):
                    if self.mapa.dentro(zf + df, zc + dc):
                        bloqueado[zf + df, zc + dc] = True
            blando = mapa_mod.dilatar(ocupado, self.inflado_blando) & ~bloqueado
            # El robot puede estar pegado a un obstaculo: liberar su entorno (nunca lo ocupado).
            f0, f1 = max(0, inicio[0] - inflado), min(self.mapa.filas, inicio[0] + inflado + 1)
            c0, c1 = max(0, inicio[1] - inflado), min(self.mapa.columnas, inicio[1] + inflado + 1)
            bloqueado[f0:f1, c0:c1] &= ocupado[f0:f1, c0:c1]
            meta = libre_mas_cercana(bloqueado, meta_celda, self.ajuste_meta)
            if meta is None:
                continue
            ruta = astar(bloqueado, inicio, meta, blando.astype(np.float32) * COSTO_BLANDO)
            if ruta is not None:
                return ruta
        return None

    def metros_restantes(self, pose):
        if not self.ruta:
            return None
        puntos = [(pose[0], pose[1])] + self.ruta[self.progreso:]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(puntos, puntos[1:]))

    def paso(self, pose, t):
        if not self.activo:
            return 0.0, 0.0, None
        x, y, yaw = pose
        if math.hypot(self.meta[0] - x, self.meta[1] - y) < TOLERANCIA_META:
            self.activo = False
            return 0.0, 0.0, LLEGO

        bloqueada = False
        if self.ruta_celdas is not None:
            ocupado = self.mapa.ocupado()
            bloqueada = any(ocupado[celda] for celda in self.ruta_celdas[self.progreso:])
        if self.ruta_celdas is None or bloqueada or t - self.ultima_planificacion >= REPLANIFICAR_CADA_S:
            self.ruta_celdas = self.planificar(pose)
            self.ultima_planificacion = t
            self.replanificaciones += 1
            if self.ruta_celdas is None:
                self.activo = False
                return 0.0, 0.0, SIN_RUTA
            self.ruta = [self.mapa.centro(f, c) for f, c in self.ruta_celdas]
            self.ruta[-1] = self.meta if math.hypot(self.ruta[-1][0] - self.meta[0], self.ruta[-1][1] - self.meta[1]) < 0.2 else self.ruta[-1]
            self.progreso = 0

        # Avanzar el indice al punto mas cercano (nunca hacia atras).
        mejor, mejor_d = self.progreso, float("inf")
        for i in range(self.progreso, min(len(self.ruta), self.progreso + 40)):
            d = math.hypot(self.ruta[i][0] - x, self.ruta[i][1] - y)
            if d < mejor_d:
                mejor, mejor_d = i, d
        self.progreso = mejor

        fin = self.ruta[-1]
        distancia_fin = math.hypot(fin[0] - x, fin[1] - y)
        if distancia_fin < TOLERANCIA_META:
            # Fin de la ruta pero no de la meta (meta ajustada a celda libre): se acerco lo posible.
            self.activo = False
            return 0.0, 0.0, LLEGO
        objetivo = fin
        for punto in self.ruta[self.progreso:]:
            if math.hypot(punto[0] - x, punto[1] - y) >= ADELANTO:
                objetivo = punto
                break
        error = angulo_normal(math.atan2(objetivo[1] - y, objetivo[0] - x) - yaw)
        omega = max(-self.omega_max, min(self.omega_max, GANANCIA_RUMBO * error))
        if abs(error) > ANGULO_GIRO_EN_SITIO:
            return 0.0, omega, None
        factor = max(FACTOR_MINIMO, math.cos(error)) * min(1.0, distancia_fin / RADIO_FRENADO)
        return self.velocidad_max * max(FACTOR_MINIMO, factor), omega, None
