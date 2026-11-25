"""Sectores del LiDAR por angulo real, en el marco del robot.

Los dos SickS300 del MiR100 van en esquinas opuestas, girados y montados boca
abajo, asi que el indice de un rayo NO dice hacia donde apunta respecto al robot.
Medido en el proyecto anterior (DistributedAI/Proyecto/mir100, lidar.py):
    frontal (esquina delantera izquierda): rayo i -> -90 + i * paso grados
    trasero (esquina trasera derecha)    : rayo i ->  90 + i * paso grados
con 0 = frente del robot, + = izquierda, paso = FOV / (rayos - 1).

Cada lectura se convierte en un punto en el marco del robot (x adelante, y a la
izquierda) y se mide su HOLGURA: distancia al borde de la huella del robot, no
al sensor. Los sectores se definen por el angulo de ese punto visto desde el
centro del robot y combinan ambos LiDAR (el derecho lo cubre sobre todo el
trasero, el izquierdo el frontal).

Sin dependencias de Webots: se prueba con tests/test_lidar.py.
"""
import math

import numpy as np

FOV_GRADOS = 270.0
PRIMER_ANGULO_FRONTAL = -90.0
PRIMER_ANGULO_TRASERO = 90.0
POSICION_FRONTAL = (0.4288, 0.2358)    # m, marco del robot (de Mir100.proto)
POSICION_TRASERO = (-0.3548, -0.2352)
MEDIO_LARGO = 0.445                    # huella del MiR100: 890 x 580 mm
MEDIO_ANCHO = 0.29
RANGO_MINIMO_VALIDO = 0.15             # mas cerca = el propio robot o ruido

# Sectores: (desde, hasta) en grados desde el centro del robot, + = izquierda.
SECTORES = {
    "front": (-25.0, 25.0),
    "front_left": (25.0, 65.0),
    "left": (65.0, 115.0),
    "front_right": (-65.0, -25.0),
    "right": (-115.0, -65.0),
    "back": (155.0, 205.0),
}
SECTORES_MINIMO = ("front", "back")    # seguridad: el punto mas cercano
PERCENTIL_LADOS = 20                   # lados: robusto a rayos sueltos
FRACCION_VALIDA_MINIMA = 0.25          # menos lecturas validas = sector desconocido
NOMBRES = ("left", "front_left", "front", "front_right", "right", "back")


def puntos_del_lidar(rangos, rango_maximo, primer_angulo, posicion, fov_grados=FOV_GRADOS):
    """Devuelve (angulos_desde_centro_grados, holguras_m, validos) de un escaneo.

    Un rayo sin eco (inf) cuenta como libre hasta rango_maximo. NaN, valores
    <= RANGO_MINIMO_VALIDO o > rango_maximo cuentan como lectura invalida.
    """
    r = np.array(rangos, dtype=float)
    total = len(r)
    if total < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
    r[np.isposinf(r)] = rango_maximo
    validos = np.isfinite(r) & (r > RANGO_MINIMO_VALIDO) & (r <= rango_maximo)
    # Un rayo invalido no tiene punto, pero si direccion: se ubica a 1 m solo
    # para saber a que sector pertenece (cuenta como lectura faltante ahi).
    r = np.where(validos, r, 1.0)
    angulos = np.radians(primer_angulo + np.arange(total) * fov_grados / (total - 1))
    x = posicion[0] + r * np.cos(angulos)
    y = posicion[1] + r * np.sin(angulos)
    # Puntos dentro de la huella: el sensor se ve a si mismo o al chasis.
    dentro = (np.abs(x) <= MEDIO_LARGO) & (np.abs(y) <= MEDIO_ANCHO)
    validos = validos & ~dentro
    holgura = np.hypot(np.maximum(np.abs(x) - MEDIO_LARGO, 0.0), np.maximum(np.abs(y) - MEDIO_ANCHO, 0.0))
    holgura = np.maximum(holgura, 0.01)  # 0.0 queda reservado para "sin datos"
    angulo_centro = np.degrees(np.arctan2(y, x))
    return angulo_centro, holgura, validos


def puntos_xy(frontal, trasero, rango_maximo_frontal, rango_maximo_trasero, fov_grados=FOV_GRADOS):
    """(xs, ys) en el marco del robot de los ecos validos de ambos LiDAR (sin los rayos sin eco)."""
    xs, ys = [], []
    for rangos, rmax, primer, pos in ((frontal, rango_maximo_frontal, PRIMER_ANGULO_FRONTAL, POSICION_FRONTAL),
                                      (trasero, rango_maximo_trasero, PRIMER_ANGULO_TRASERO, POSICION_TRASERO)):
        r = np.array(rangos, dtype=float)
        if len(r) < 2:
            continue
        a = np.radians(primer + np.arange(len(r)) * fov_grados / (len(r) - 1))
        eco = np.isfinite(r) & (r > RANGO_MINIMO_VALIDO) & (r < rmax * 0.999)
        x = pos[0] + r[eco] * np.cos(a[eco])
        y = pos[1] + r[eco] * np.sin(a[eco])
        fuera = ~((np.abs(x) <= MEDIO_LARGO) & (np.abs(y) <= MEDIO_ANCHO))
        xs.append(x[fuera])
        ys.append(y[fuera])
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)


def _en_sector(angulos, desde, hasta):
    # Angulos en (-180, 180]; los sectores pueden cruzar 180 (atras).
    a = np.mod(angulos - desde, 360.0)
    return a <= (hasta - desde)


def distancias_por_sector(frontal, trasero, rango_maximo_frontal, rango_maximo_trasero, fov_grados=FOV_GRADOS):
    """Holgura por sector (m) combinando ambos LiDAR. 0.0 = sector desconocido.

    Devuelve el mismo formato que usa el controlador: left, front_left, front,
    front_right, right, back y ok (todos los sectores con datos).
    """
    a1, h1, v1 = puntos_del_lidar(frontal, rango_maximo_frontal, PRIMER_ANGULO_FRONTAL, POSICION_FRONTAL, fov_grados)
    a2, h2, v2 = puntos_del_lidar(trasero, rango_maximo_trasero, PRIMER_ANGULO_TRASERO, POSICION_TRASERO, fov_grados)
    angulos = np.concatenate([a1, a2])
    holguras = np.concatenate([h1, h2])
    validos = np.concatenate([v1, v2])

    datos = {}
    for nombre in NOMBRES:
        desde, hasta = SECTORES[nombre]
        en_sector = _en_sector(angulos, desde, hasta)
        total = int(np.count_nonzero(en_sector))
        buenos = holguras[en_sector & validos]
        if total == 0 or len(buenos) < max(1, math.ceil(total * FRACCION_VALIDA_MINIMA)):
            datos[nombre] = 0.0
        elif nombre in SECTORES_MINIMO:
            datos[nombre] = float(np.min(buenos))
        else:
            datos[nombre] = float(np.percentile(buenos, PERCENTIL_LADOS))
    datos["ok"] = all(datos[n] > 0.0 for n in NOMBRES)
    return datos
