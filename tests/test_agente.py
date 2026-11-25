"""Tests del agente autonomo sin llamar a la API: robot simulado + servidor MCP real
(por stdio) + un tool runner falso que imita las decisiones del modelo.

Requiere el entorno del agente (anthropic>=2.14):
    .venv/bin/python -m unittest tests.test_agente -v
"""
import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "agente"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import agente
    DISPONIBLE = True
except ImportError:  # sin anthropic 2.x (python del sistema): se omite
    DISPONIBLE = False

from test_herramientas_mcp import RobotSimulado  # noqa: E402


def bloque_herramienta(nombre, entrada, n):
    return SimpleNamespace(type="tool_use", name=nombre, input=entrada, id="toolu_{}".format(n))


def mensaje(bloques, stop="tool_use"):
    return SimpleNamespace(content=bloques, stop_reason=stop, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


class RunnerFalso:
    """Ejecuta un guion de llamadas como lo haria el tool runner: produce el mensaje y,
    al pedir el siguiente, ejecuta las herramientas del anterior."""

    def __init__(self, guion, herramientas, max_iterations):
        self.guion = guion
        self.herramientas = {h.to_dict()["name"]: h for h in herramientas}
        self.max_iterations = max_iterations
        self.resultados = []

    def __aiter__(self):
        return self._iterar()

    async def _iterar(self):
        for i, bloques in enumerate(self.guion[: self.max_iterations]):
            msj = mensaje(bloques, "tool_use" if any(b.type == "tool_use" for b in bloques) else "end_turn")
            yield msj
            for b in bloques:
                if b.type == "tool_use":
                    self.resultados.append((b.name, await self.herramientas[b.name].call(b.input)))


class ClienteFalso:
    def __init__(self, guion):
        self.guion = guion
        self.runner = None
        self.parametros = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs):
        self.parametros = kwargs
        self.runner = RunnerFalso(self.guion, kwargs["tools"], kwargs["max_iterations"])
        return self.runner


@unittest.skipUnless(DISPONIBLE, "requiere el .venv del agente (anthropic 2.x)")
class TestAgente(unittest.TestCase):
    def setUp(self):
        self.robot = RobotSimulado()
        os.environ["MIR_PUENTE_PUERTO"] = str(self.robot.puente.puerto)
        self.original = agente.anthropic.AsyncAnthropic

    def tearDown(self):
        agente.anthropic.AsyncAnthropic = self.original
        self.robot.cerrar()

    def correr(self, guion, **opciones):
        cliente = ClienteFalso(guion)
        agente.anthropic.AsyncAnthropic = lambda: cliente
        args = {"max_pasos": 12, "max_movimientos": 5, "minutos": 5, "esfuerzo": "high", **opciones}
        mision = asyncio.run(agente.ejecutar("Vaya al punto de entrega 3", **args))
        return mision, cliente

    def test_exito_reportado_por_el_agente(self):
        guion = [
            [bloque_herramienta("estado_robot", {}, 1)],
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_2"}, 2)],
            [bloque_herramienta("reportar_resultado", {"resultado": "exito", "resumen": "Llegue a la entrega 2.",
                                                       "pasos_realizados": ["ir_a ENTREGA_2"]}, 3)],
            [SimpleNamespace(type="text", text="Listo.")],
        ]
        mision, cliente = self.correr(guion)
        self.assertEqual(mision.reporte["resultado"], "exito")
        self.assertEqual(mision.reporte["generado_por"], "agente")
        self.assertEqual(mision.movimientos, 1)
        self.assertNotIn(("parar", {}), self.robot.ordenes)  # exito: no se frena el robot
        p = cliente.parametros
        self.assertEqual(p["model"], "claude-opus-5-5")
        self.assertEqual(p["thinking"], {"type": "adaptive"})
        self.assertEqual(p["fallbacks"], "default")
        nombres = sorted(h.to_dict()["name"] for h in p["tools"])
        self.assertIn("reportar_resultado", nombres)
        self.assertIn("ir_a", nombres)

    def test_fallo_y_recuperacion_reportados(self):
        guion = [
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 1)],
            [bloque_herramienta("escanear_entorno", {}, 2)],
            [bloque_herramienta("avanzar", {"metros": -0.4}, 3)],
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 4)],
            [bloque_herramienta("parar", {}, 5)],
            [bloque_herramienta("reportar_resultado", {"resultado": "fallo", "resumen": "La ruta esta bloqueada.",
                                                       "motivo": "evasion_sin_salida", "pasos_realizados": ["..."]}, 6)],
            [SimpleNamespace(type="text", text="No pude.")],
        ]
        mision, cliente = self.correr(guion)
        self.assertEqual((mision.reporte["resultado"], mision.reporte["motivo"]), ("fallo", "evasion_sin_salida"))
        self.assertEqual(mision.ultimo_resultado_robot["motivo"], "evasion_sin_salida")
        self.assertEqual(mision.movimientos, 3)
        self.assertIn(("parar", {}), self.robot.ordenes)

    def test_sin_reporte_genera_respaldo_y_frena(self):
        guion = [[bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, n)] for n in range(10)]
        mision, _ = self.correr(guion, max_pasos=3)
        self.assertEqual(mision.reporte["generado_por"], "programa")
        self.assertEqual(mision.reporte["motivo"], "limite_de_pasos")
        self.assertIn("retroceder", mision.reporte["recomendacion"])
        self.assertEqual(mision.reporte["ultimo_resultado_del_robot"]["motivo"], "evasion_sin_salida")
        self.assertIn(("parar", {}), self.robot.ordenes)

    def test_sin_progreso_detiene_tras_3_intentos(self):
        guion = [[bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, n)] for n in range(6)]
        mision, cliente = self.correr(guion, max_movimientos=30)
        # 1er fallo fija la mejor distancia; 3 mas sin acercarse agotan el limite y el 5o se rechaza.
        self.assertEqual(mision.movimientos, 4)
        self.assertTrue(mision.sin_progreso())
        self.assertIn("sin acercarse", cliente.runner.resultados[4][1])
        self.assertEqual(mision.reporte["motivo"], "sin_progreso_hacia_el_destino")
        self.assertIn(("parar", {}), self.robot.ordenes)

    def test_progreso_permite_seguir(self):
        guion = [
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 1)],
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 2)],
            [bloque_herramienta("ir_a_punto", {"x": 1.0, "y": 4.5}, 3)],   # se acerca 2.4 m
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 4)],
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 5)],
            [bloque_herramienta("ir_a_punto", {"x": -2.0, "y": 4.4}, 6)],  # se acerca 3 m mas
            [bloque_herramienta("ir_a", {"destino": "ENTREGA_3"}, 7)],
        ]
        mision, cliente = self.correr(guion, max_movimientos=30)
        self.assertEqual(mision.movimientos, 7)  # ninguno rechazado
        self.assertFalse(mision.sin_progreso())
        self.assertLess(mision.mejor_distancia, 2.0)
        texto = agente.texto_de_resultado(cliente.runner.resultados[-1][1])
        self.assertIn("Progreso: a", texto)

    def test_limite_de_movimientos(self):
        guion = [[bloque_herramienta("girar", {"grados": 10}, n)] for n in range(4)]
        mision, cliente = self.correr(guion, max_movimientos=2)
        self.assertEqual(mision.movimientos, 2)
        ultimo = cliente.runner.resultados[-1][1]
        self.assertIn("Limite de 2 ordenes de movimiento", ultimo)

    def test_sin_robot_no_llama_al_modelo(self):
        self.robot.cerrar()
        os.environ["MIR_PUENTE_PUERTO"] = "1"
        mision, cliente = self.correr([[bloque_herramienta("estado_robot", {}, 1)]])
        self.assertIsNone(cliente.parametros)
        self.assertEqual(mision.reporte["motivo"], "sin_conexion_con_el_robot")
        self.robot = RobotSimulado()  # para tearDown

    def test_reporte_rechaza_resultado_invalido(self):
        mision = agente.Mision("x", 1, 60)
        herramienta = agente.crear_herramienta_reporte(mision)
        respuesta = asyncio.run(herramienta.call({"resultado": "quizas", "resumen": "", "pasos_realizados": []}))
        self.assertIn("Error", agente.texto_de_resultado(respuesta))
        self.assertIsNone(mision.reporte)


if __name__ == "__main__":
    unittest.main()
