"""Tests de los sectores del LiDAR por angulo real, sin Webots.

Se simulan los dos SickS300 trazando rayos contra paredes ubicadas en el marco
del robot, con la misma convencion de montaje que lidar.py.
"""
import math
import os
import sys
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))

import lidar  # noqa: E402

RAYOS = 541
RANGO_MAXIMO = 30.0


def escaneo(primer_angulo, posicion, paredes):
    """paredes: lista de ('x', c) o ('y', c): rectas x=c o y=c en el marco del robot."""
    rangos = []
    for i in range(RAYOS):
        a = math.radians(primer_angulo + i * lidar.FOV_GRADOS / (RAYOS - 1))
        dx, dy = math.cos(a), math.sin(a)
        mejor = float("inf")
        for eje, c in paredes:
            componente, origen = (dx, posicion[0]) if eje == "x" else (dy, posicion[1])
            if abs(componente) > 1e-9:
                t = (c - origen) / componente
                if 0 < t < mejor:
                    mejor = t
        rangos.append(mejor if mejor <= RANGO_MAXIMO else float("inf"))
    return rangos


def sectores(paredes):
    frontal = escaneo(lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL, paredes)
    trasero = escaneo(lidar.PRIMER_ANGULO_TRASERO, lidar.POSICION_TRASERO, paredes)
    return lidar.distancias_por_sector(frontal, trasero, RANGO_MAXIMO, RANGO_MAXIMO)


class TestSectoresLidar(unittest.TestCase):
    def test_pared_al_frente(self):
        s = sectores([("x", lidar.MEDIO_LARGO + 1.0)])
        self.assertAlmostEqual(s["front"], 1.0, places=2)
        self.assertGreater(s["left"], 3.0)
        self.assertGreater(s["right"], 3.0)
        self.assertGreater(s["back"], 20.0)

    def test_pared_a_la_derecha_no_se_confunde_con_la_izquierda(self):
        # El error original: con cortes por porcentaje, la derecha se leia como "left".
        s = sectores([("y", -(lidar.MEDIO_ANCHO + 0.5))])
        self.assertAlmostEqual(s["right"], 0.5, delta=0.05)
        self.assertGreater(s["left"], 5.0)
        self.assertLess(s["front_right"], s["front_left"])

    def test_pared_a_la_izquierda(self):
        s = sectores([("y", lidar.MEDIO_ANCHO + 0.5)])
        self.assertAlmostEqual(s["left"], 0.5, delta=0.05)
        self.assertGreater(s["right"], 5.0)
        self.assertLess(s["front_left"], s["front_right"])

    def test_pared_detras(self):
        s = sectores([("x", -(lidar.MEDIO_LARGO + 0.8))])
        self.assertAlmostEqual(s["back"], 0.8, places=2)
        self.assertGreater(s["front"], 20.0)

    def test_pasillo(self):
        s = sectores([("y", lidar.MEDIO_ANCHO + 0.7), ("y", -(lidar.MEDIO_ANCHO + 0.9)), ("x", lidar.MEDIO_LARGO + 2.0)])
        self.assertAlmostEqual(s["left"], 0.7, delta=0.05)
        self.assertAlmostEqual(s["right"], 0.9, delta=0.05)
        # El sector frontal (+-25 grados) ve las paredes laterales solo lejos, cerca de la del fondo.
        self.assertGreater(s["front"], 1.5)
        self.assertLessEqual(s["front"], 2.0 + 1e-6)
        self.assertTrue(s["ok"])

    def test_convencion_del_rayo_frontal(self):
        # Rayo i del frontal apunta a -90 + i * paso: un eco solo en el rayo de 0 grados cae al frente.
        frontal = [float("nan")] * RAYOS
        paso = lidar.FOV_GRADOS / (RAYOS - 1)
        frontal[int(round((0.0 - lidar.PRIMER_ANGULO_FRONTAL) / paso))] = 1.0
        a, h, v = lidar.puntos_del_lidar(frontal, RANGO_MAXIMO, lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL)
        self.assertEqual(int(v.sum()), 1)
        indice = int(v.argmax())
        self.assertTrue(-25 < a[indice] < 25)

    def test_sin_datos_es_desconocido(self):
        nada = [float("nan")] * RAYOS
        s = lidar.distancias_por_sector(nada, nada, RANGO_MAXIMO, RANGO_MAXIMO)
        self.assertFalse(s["ok"])
        self.assertEqual(s["front"], 0.0)

    def test_sin_eco_es_libre(self):
        libre = [float("inf")] * RAYOS
        s = lidar.distancias_por_sector(libre, libre, RANGO_MAXIMO, RANGO_MAXIMO)
        self.assertTrue(s["ok"])
        self.assertGreater(s["front"], 25.0)


if __name__ == "__main__":
    unittest.main()
