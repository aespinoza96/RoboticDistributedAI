"""Tests de la ubicacion de personas y de la zona que el planificador rodea, sin Webots."""
import math
import os
import sys
import unittest

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mapa  # noqa: E402
import personas  # noqa: E402
import planificador  # noqa: E402
from test_planificador import simular  # noqa: E402

ANCHO, ALTO, FOV = 640, 480, 1.5708


def caja_para(angulo_grados, distancia, ancho_m=0.5, alto_m=personas.ALTURA_PERSONA):
    """Recuadro de una persona a cierta direccion y distancia de la camara."""
    f = (ANCHO / 2.0) / math.tan(FOV / 2.0)
    cx = (ANCHO - 1) / 2.0 - f * math.tan(math.radians(angulo_grados))
    medio_ancho = f * ancho_m / 2.0 / distancia
    medio_alto = f * alto_m / 2.0 / distancia
    return (cx - medio_ancho, ALTO / 2.0 - medio_alto, cx + medio_ancho, ALTO / 2.0 + medio_alto)


def puntos_persona(angulo_grados, distancia, n=40):
    a = math.radians(angulo_grados)
    xs = personas.POSICION_CAMARA_X + distancia * math.cos(a) + np.random.default_rng(0).normal(0, 0.01, n)
    ys = distancia * math.sin(a) + np.linspace(-0.2, 0.2, n)
    return xs, ys


class TestUbicar(unittest.TestCase):
    def test_persona_al_frente_con_profundidad(self):
        p = personas.ubicar(caja_para(0, 1.0), ANCHO, ALTO, FOV, puntos_persona(0, 1.0))
        self.assertEqual(p["fuente"], "profundidad")
        self.assertAlmostEqual(p["distancia"], 1.0, delta=0.15)
        self.assertAlmostEqual(p["angulo"], 0.0, delta=1.0)
        self.assertAlmostEqual(p["holgura"], personas.POSICION_CAMARA_X + p["distancia"] - 0.445, delta=0.05)

    def test_persona_a_la_izquierda(self):
        p = personas.ubicar(caja_para(30, 1.5), ANCHO, ALTO, FOV, puntos_persona(30, 1.5))
        self.assertAlmostEqual(p["angulo"], 30.0, delta=2.0)
        self.assertGreater(p["y"], 0.5)

    def test_sin_profundidad_estima_por_tamano(self):
        p = personas.ubicar(caja_para(-20, 4.0), ANCHO, ALTO, FOV, None)
        self.assertEqual(p["fuente"], "tamano")
        self.assertAlmostEqual(p["distancia"], 4.0, delta=0.1)
        self.assertAlmostEqual(p["angulo"], -20.0, delta=1.0)

    def test_puntos_fuera_del_recuadro_se_ignoran(self):
        # Hay una pared a la derecha (puntos), pero la persona esta a la izquierda: no se usa la pared.
        pared = (np.full(30, 1.2), np.linspace(-1.5, -1.0, 30))
        p = personas.ubicar(caja_para(25, 3.0), ANCHO, ALTO, FOV, pared)
        self.assertEqual(p["fuente"], "tamano")


class TestZonas(unittest.TestCase):
    def test_la_ruta_rodea_a_la_persona(self):
        m = mapa.Mapa()
        nav = planificador.Navegador(m, 0.3, 0.3)
        nav.iniciar((5.0, 0.0))
        nav.fijar_zonas([(2.5, 0.0, 1.0)])               # PERSONA_ZONA_RADIO del controlador
        resultado, _, recorrido = simular(nav, (0.0, 0.0, 0.0))
        self.assertEqual(resultado, planificador.LLEGO)
        self.assertTrue(all(math.hypot(x - 2.5, y) > 0.8 for x, y in recorrido))

    def test_persona_nueva_fuerza_replanificar(self):
        m = mapa.Mapa()
        nav = planificador.Navegador(m, 0.3, 0.3)
        nav.iniciar((5.0, 0.0))
        simular(nav, (0.0, 0.0, 0.0), pasos=3)
        self.assertIsNotNone(nav.ruta_celdas)
        nav.fijar_zonas([(2.5, 0.0, 0.8)])
        self.assertIsNone(nav.ruta_celdas)
        nav.fijar_zonas([(2.6, 0.05, 0.8)])        # casi la misma: no replanifica de nuevo
        simular(nav, (0.0, 0.0, 0.0), pasos=1)
        antes = nav.replanificaciones
        nav.fijar_zonas([(2.6, 0.05, 0.8)])
        simular(nav, (0.0, 0.0, 0.0), pasos=1)
        self.assertEqual(nav.replanificaciones, antes)


if __name__ == "__main__":
    unittest.main()
