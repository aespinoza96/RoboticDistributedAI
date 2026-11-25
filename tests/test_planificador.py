"""Tests del mapa de ocupacion, A* y el navegador, sin Webots.

El escenario principal imita la fabrica: una cinta transportadora larga corta el paso
entre el robot y la entrega, y solo se puede pasar rodeandola por un extremo.
"""
import math
import os
import sys
import unittest

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lidar  # noqa: E402
import mapa  # noqa: E402
import planificador  # noqa: E402
from test_lidar import RAYOS  # noqa: E402


def marcar_rectangulo(m, x0, x1, y0, y1, capa="log_odds"):
    f0, c0 = m.celda(x0, y0)
    f1, c1 = m.celda(x1, y1)
    getattr(m, capa)[max(0, f0):f1 + 1, max(0, c0):c1 + 1] = 3.0


def simular(navegador, pose, pasos=4000, dt=0.05):
    """Integra un uniciclo con las ordenes del navegador. Devuelve (resultado, pose, recorrido)."""
    x, y, yaw = pose
    recorrido = [(x, y)]
    for i in range(pasos):
        v, w, resultado = navegador.paso((x, y, yaw), i * dt)
        if resultado is not None:
            return resultado, (x, y, yaw), recorrido
        yaw += w * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        recorrido.append((x, y))
    return None, (x, y, yaw), recorrido


class TestAStar(unittest.TestCase):
    def test_rodea_un_muro_por_el_hueco(self):
        bloqueado = np.zeros((20, 20), dtype=bool)
        bloqueado[10, :17] = True                 # muro con hueco a la derecha
        ruta = planificador.astar(bloqueado, (2, 2), (18, 2))
        self.assertIsNotNone(ruta)
        self.assertTrue(any(c >= 17 for f, c in ruta if f == 10))

    def test_sin_hueco_no_hay_ruta(self):
        bloqueado = np.zeros((20, 20), dtype=bool)
        bloqueado[10, :] = True
        self.assertIsNone(planificador.astar(bloqueado, (2, 2), (18, 2)))

    def test_meta_en_obstaculo_se_ajusta_a_libre(self):
        bloqueado = np.zeros((10, 10), dtype=bool)
        bloqueado[4:7, 4:7] = True
        celda = planificador.libre_mas_cercana(bloqueado, (5, 5), 5)
        self.assertFalse(bloqueado[celda])


class TestMapa(unittest.TestCase):
    def test_lidar_marca_pared_y_espacio_libre(self):
        m = mapa.Mapa()
        rangos = []
        pared = 3.0  # pared en x = 3 m del centro del robot, robot en el origen mirando a +x
        for i in range(RAYOS):
            a = math.radians(lidar.PRIMER_ANGULO_FRONTAL + i * lidar.FOV_GRADOS / (RAYOS - 1))
            px, py = lidar.POSICION_FRONTAL
            rangos.append((pared - px) / math.cos(a) if math.cos(a) > 0.2 else float("inf"))
        for _ in range(3):
            m.actualizar_lidar((0.0, 0.0, 0.0), rangos, lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL, 30.0)
        self.assertTrue(m.ocupado()[m.celda(3.0, 0.3)])
        self.assertFalse(m.ocupado()[m.celda(1.5, 0.0)])
        self.assertLess(m.log_odds[m.celda(1.5, 0.0)], 0)

    def test_el_lidar_no_borra_una_mesa_vista_por_la_profundidad(self):
        m = mapa.Mapa()
        xs, ys = np.full(30, 1.5), np.linspace(-0.5, 0.5, 30)   # tabla de mesa a 1.5 m
        m.actualizar_alto((0.0, 0.0, 0.0), xs, ys)
        m.actualizar_alto((0.0, 0.0, 0.0), xs, ys)
        libres = [float("inf")] * RAYOS                          # el LiDAR pasa por debajo
        for _ in range(5):
            m.actualizar_lidar((0.0, 0.0, 0.0), libres, lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL, 30.0)
        self.assertTrue(m.ocupado()[m.celda(1.5, 0.0)])

    def test_impacto_lejano_no_marca_ocupado(self):
        m = mapa.Mapa()
        rangos = [float("inf")] * RAYOS
        i = int(round((0.0 - lidar.PRIMER_ANGULO_FRONTAL) / (lidar.FOV_GRADOS / (RAYOS - 1))))
        rangos[i] = 5.5                                           # un punto suelto a 5.5 m
        for _ in range(5):
            m.actualizar_lidar((0.0, 0.0, 0.0), rangos, lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL, 30.0)
        x = lidar.POSICION_FRONTAL[0] + 5.5
        self.assertFalse(m.ocupado()[m.celda(x, lidar.POSICION_FRONTAL[1])])
        self.assertLess(m.log_odds[m.celda(3.0, lidar.POSICION_FRONTAL[1])], 0)  # el tramo previo si queda libre

    def test_pose_girada(self):
        m = mapa.Mapa()
        m.actualizar_alto((2.0, 1.0, math.pi / 2), np.full(5, 1.0), np.zeros(5))  # 1 m adelante, mirando a +y
        m.actualizar_alto((2.0, 1.0, math.pi / 2), np.full(5, 1.0), np.zeros(5))
        self.assertTrue(m.ocupado()[m.celda(2.0, 2.0)])


class TestNavegador(unittest.TestCase):
    def fabrica(self):
        m = mapa.Mapa()
        marcar_rectangulo(m, -0.8, 7.2, -1.4, -0.7)              # la cinta de 8 m
        return m

    def test_rodea_la_cinta_para_llegar_a_la_entrega(self):
        m = self.fabrica()
        nav = planificador.Navegador(m, 0.3, 0.3)
        nav.iniciar((3.33, -3.78))                                # ENTREGA_1
        resultado, pose, recorrido = simular(nav, (3.36, 4.95, -math.pi / 2))
        self.assertEqual(resultado, planificador.LLEGO)
        self.assertLess(math.hypot(pose[0] - 3.33, pose[1] + 3.78), planificador.TOLERANCIA_META + 0.05)
        # Cruzo la linea de la cinta por fuera de sus extremos, nunca a traves de ella.
        cruces = [x for (x, y), (_, y2) in zip(recorrido, recorrido[1:]) if (y - (-1.05)) * (y2 - (-1.05)) <= 0]
        self.assertTrue(cruces and all(x > 7.2 + 0.3 or x < -0.8 - 0.3 for x in cruces))

    def test_sin_ruta_si_todo_esta_cerrado(self):
        m = mapa.Mapa()
        marcar_rectangulo(m, -10, 10, -1.4, -0.7)                 # muro de lado a lado
        nav = planificador.Navegador(m, 0.3, 0.3)
        nav.iniciar((3.33, -3.78))
        resultado, _, _ = simular(nav, (3.36, 4.95, 0.0), pasos=10)
        self.assertEqual(resultado, planificador.SIN_RUTA)

    def test_replanifica_si_aparece_un_obstaculo(self):
        m = mapa.Mapa()
        nav = planificador.Navegador(m, 0.3, 0.3)
        nav.iniciar((4.0, 0.0))
        simular(nav, (0.0, 0.0, 0.0), pasos=5)
        antes = nav.replanificaciones
        marcar_rectangulo(m, 1.8, 2.2, -0.6, 0.6)                 # aparece algo en medio de la ruta
        resultado, pose, recorrido = simular(nav, (0.3, 0.0, 0.0))
        self.assertGreater(nav.replanificaciones, antes)
        self.assertEqual(resultado, planificador.LLEGO)
        self.assertFalse(any(1.8 <= x <= 2.2 and -0.6 <= y <= 0.6 for x, y in recorrido))


if __name__ == "__main__":
    unittest.main()
