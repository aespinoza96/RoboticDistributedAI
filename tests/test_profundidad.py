"""Tests de la camara de profundidad, sin Webots: se renderiza una imagen de
profundidad sintetica (suelo, tabla de mesa, pared) con el mismo modelo de camara."""
import math
import os
import sys
import unittest

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))

import lidar  # noqa: E402
import profundidad  # noqa: E402

FOV = 2.05949        # Mir100.proto
ANCHO, ALTO = 128, 72
RANGO_MIN, RANGO_MAX = 0.18, 1.95
CX, CY, CZ = profundidad.POSICION_CAMARA


def renderizar(mesa=None, pared_x=None):
    """Profundidad plana por pixel. mesa = (x0, x1, y0, y1, altura) en el marco del robot."""
    focal = (ANCHO / 2.0) / math.tan(FOV / 2.0)
    imagen = np.full((ALTO, ANCHO), np.inf)
    for v in range(ALTO):
        for u in range(ANCHO):
            du = (u - (ANCHO - 1) / 2.0) / focal
            dv = (v - (ALTO - 1) / 2.0) / focal
            candidatos = []
            if dv > 0:
                candidatos.append(CZ / dv)  # suelo
            if mesa is not None and dv < 0:
                x0, x1, y0, y1, h = mesa
                d = (h - CZ) / -dv
                x, y = CX + d, CY - du * d
                if x0 <= x <= x1 and y0 <= y <= y1:
                    candidatos.append(d)
            if pared_x is not None:
                candidatos.append(pared_x - CX)
            d = min([c for c in candidatos if c > 0], default=np.inf)
            imagen[v, u] = d if d < RANGO_MAX else np.inf
    return imagen


def sectores(imagen):
    return profundidad.holgura_por_sector(*profundidad.puntos_obstaculo(imagen, FOV, RANGO_MIN, RANGO_MAX, salto=1))


class TestProfundidad(unittest.TestCase):
    def test_solo_suelo_no_es_obstaculo(self):
        self.assertEqual(sectores(renderizar()), {})

    def test_tabla_de_mesa_al_frente(self):
        borde = lidar.MEDIO_LARGO + 0.6
        s = sectores(renderizar(mesa=(borde, borde + 0.8, -0.9, 0.9, 0.70)))
        holgura, altura = s["front"]
        self.assertAlmostEqual(holgura, 0.6, delta=0.06)
        self.assertAlmostEqual(altura, 0.70, delta=0.01)

    def test_mesa_mas_alta_que_el_robot_se_ignora(self):
        borde = lidar.MEDIO_LARGO + 0.6
        self.assertEqual(sectores(renderizar(mesa=(borde, borde + 0.8, -0.9, 0.9, 1.20))), {})

    def test_pared_al_frente(self):
        s = sectores(renderizar(pared_x=lidar.MEDIO_LARGO + 1.0))
        self.assertAlmostEqual(s["front"][0], 1.0, delta=0.03)

    def test_combinar_mesa_que_el_lidar_no_ve(self):
        datos_lidar = {"left": 2.0, "front_left": 2.5, "front": 3.0, "front_right": 2.5, "right": 2.0, "back": 1.0, "ok": True}
        combinado, avisos = profundidad.combinar(datos_lidar, {"front": (0.6, 0.7)})
        self.assertEqual(combinado["front"], 0.6)
        self.assertEqual(combinado["left"], 2.0)
        self.assertEqual([a["sector"] for a in avisos], ["front"])

    def test_combinar_pared_vista_por_ambos_no_avisa(self):
        combinado, avisos = profundidad.combinar({"front": 1.0, "ok": True}, {"front": (0.95, 0.3)})
        self.assertEqual(combinado["front"], 0.95)
        self.assertEqual(avisos, [])

    def test_combinar_no_cambia_un_sector_desconocido(self):
        combinado, _ = profundidad.combinar({"front": 0.0, "ok": False}, {"front": (0.5, 0.7)})
        self.assertEqual(combinado["front"], 0.0)


if __name__ == "__main__":
    unittest.main()
