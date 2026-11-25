"""Obstaculos 3D desde la camara de profundidad del MiR100.

El LiDAR escanea un solo plano a ~0,19 m del suelo: no ve la tabla de una mesa,
una cinta elevada ni nada por encima de ese plano. La camara de profundidad
(RangeFinder "depth_camera" de Mir100.proto: x=0.464, z=0.18, mirando al frente,
118 grados de ancho, 0.18-1.95 m) si.

Cada pixel se proyecta a un punto 3D en el marco del robot (x adelante, y a la
izquierda, z hacia arriba desde el suelo). Son obstaculo los puntos entre
ALTURA_SUELO y ALTURA_ROBOT (lo que el robot, con su carga, puede chocar). Igual
que en lidar.py, cada punto da un angulo desde el centro del robot y una holgura
al borde de su huella, para combinarlo con los sectores del LiDAR.

Sin dependencias de Webots: se prueba con tests/test_profundidad.py.
"""
import numpy as np

import lidar

POSICION_CAMARA = (0.464, 0.0, 0.18)   # m, marco del robot (de Mir100.proto)
ALTURA_SUELO = 0.10                    # por debajo: suelo (y las marcas de entrega de 7 cm, que se pisan)
ALTURA_ROBOT = 1.00                    # MiR100 + Robotino + cinta: margen conservador
SALTO_PIXELES = 6                      # submuestreo: 1280x720 -> ~214x120 puntos
PUNTOS_MINIMOS = 3                     # un sector necesita al menos estos puntos
MARGEN_AVISO = 0.15                    # m: cuanto mas cerca que el LiDAR para reportarlo como obstaculo alto
PROFUNDIDAD_PLANA = True               # Webots RangeFinder: distancia al plano de la camara (verificado en Webots)


def puntos_obstaculo_xyz(imagen, fov_horizontal, rango_minimo, rango_maximo, salto=SALTO_PIXELES, plana=PROFUNDIDAD_PLANA):
    """Puntos (x, y, z) en el marco del robot que el robot podria chocar."""
    d = np.asarray(imagen, dtype=float)[::salto, ::salto]
    alto_original = np.asarray(imagen).shape[0]
    ancho_original = np.asarray(imagen).shape[1]
    filas = np.arange(0, alto_original, salto, dtype=float)[: d.shape[0]]
    columnas = np.arange(0, ancho_original, salto, dtype=float)[: d.shape[1]]
    v, u = np.meshgrid(filas, columnas, indexing="ij")
    focal = (ancho_original / 2.0) / np.tan(fov_horizontal / 2.0)
    du = (u - (ancho_original - 1) / 2.0) / focal     # + = derecha en la imagen
    dv = (v - (alto_original - 1) / 2.0) / focal      # + = abajo en la imagen
    validos = np.isfinite(d) & (d >= rango_minimo) & (d < rango_maximo * 0.999)
    if plana:
        adelante = d
    else:
        adelante = d / np.sqrt(1.0 + du * du + dv * dv)
    x = POSICION_CAMARA[0] + adelante
    y = POSICION_CAMARA[1] - du * adelante
    z = POSICION_CAMARA[2] - dv * adelante
    obstaculo = validos & (z > ALTURA_SUELO) & (z < ALTURA_ROBOT)
    return x[obstaculo], y[obstaculo], z[obstaculo]


def holguras_de_puntos(x, y, z):
    """(angulos_grados, holguras_m, alturas_m) de puntos en el marco del robot."""
    holgura = np.hypot(np.maximum(np.abs(x) - lidar.MEDIO_LARGO, 0.0), np.maximum(np.abs(y) - lidar.MEDIO_ANCHO, 0.0))
    holgura = np.maximum(holgura, 0.01)
    return np.degrees(np.arctan2(y, x)), holgura, z


def puntos_obstaculo(imagen, fov_horizontal, rango_minimo, rango_maximo, salto=SALTO_PIXELES, plana=PROFUNDIDAD_PLANA):
    """imagen: array alto x ancho de distancias (m). Devuelve (angulos_grados, holguras_m, alturas_m)
    solo de los puntos que el robot podria chocar."""
    return holguras_de_puntos(*puntos_obstaculo_xyz(imagen, fov_horizontal, rango_minimo, rango_maximo, salto, plana))


def holgura_por_sector(angulos, holguras, alturas):
    """{sector: (holgura_minima, altura_del_punto)} para los sectores con suficientes puntos."""
    resultado = {}
    for nombre, (desde, hasta) in lidar.SECTORES.items():
        en_sector = np.mod(angulos - desde, 360.0) <= (hasta - desde)
        if np.count_nonzero(en_sector) < PUNTOS_MINIMOS:
            continue
        h = holguras[en_sector]
        orden = np.argsort(h)
        # El PUNTOS_MINIMOS-esimo mas cercano: ignora pixeles sueltos.
        k = orden[min(PUNTOS_MINIMOS, len(orden)) - 1]
        resultado[nombre] = (float(h[k]), float(alturas[en_sector][k]))
    return resultado


def combinar(datos_lidar, por_sector):
    """Cada sector queda con la menor holgura entre LiDAR y profundidad.

    Nunca vuelve 'desconocido' un sector: la profundidad solo agrega obstaculos.
    Devuelve los datos combinados y la lista de sectores donde la profundidad vio
    algo mas cerca que el LiDAR (normalmente, algo por encima de su plano).
    """
    combinado = dict(datos_lidar)
    mas_cerca = []
    for nombre, (holgura, altura) in por_sector.items():
        valor_lidar = combinado.get(nombre, 0.0)
        if valor_lidar > 0.0 and holgura < valor_lidar:
            combinado[nombre] = holgura
            # Solo se informa si la diferencia es clara: una pared la ven los dos casi igual.
            if valor_lidar - holgura > MARGEN_AVISO:
                mas_cerca.append({"sector": nombre, "holgura": round(holgura, 2), "altura": round(altura, 2),
                                  "lidar": round(valor_lidar, 2)})
    return combinado, mas_cerca
