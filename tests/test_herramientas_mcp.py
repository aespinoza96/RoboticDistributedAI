"""Tests de las herramientas del agente y del servidor MCP contra un robot falso, sin Webots.

    python3 -m unittest discover -s tests -v
"""
import asyncio
import json
import os
import sys
import threading
import time
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))
sys.path.insert(0, os.path.join(RAIZ, "agente"))

import herramientas  # noqa: E402
from puente import PuenteComandos  # noqa: E402


class RobotSimulado:
    """Imita al controlador: tareas con eventos y un final con motivo.

    - ir_a ENTREGA_2: llega (exito / llego).
    - ir_a ENTREGA_3: evade y falla con evasion_sin_salida.
    - girar / avanzar: maniobra_completa.
    """

    def __init__(self):
        self.puente = PuenteComandos(puerto=0)
        self.t = 0.0
        self.contador = 0
        self.tarea = None
        self.ultima = None
        self.pendientes = []  # (instante_real, evento, datos)
        self.ordenes = []
        self.posicion = [3.36, 4.95]
        self.motivo_e3 = "evasion_sin_salida"
        self.estado_e3 = "fallo"
        # (metros_por_ruta o None, mapa_conocido_m2) para el comando "ruta"; por defecto linea recta.
        self.ruta_fn = lambda pos, meta: (((pos[0] - meta[0]) ** 2 + (pos[1] - meta[1]) ** 2) ** 0.5, 50.0)
        self.activo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.activo:
            self.t += 0.1
            for cliente, mensaje in self.puente.poll():
                self._atender(cliente, mensaje)
            ahora = time.monotonic()
            for item in [p for p in self.pendientes if p[0] <= ahora]:
                self.pendientes.remove(item)
                _, evento, datos = item
                if evento == "tarea_terminada":
                    self.tarea, self.ultima = None, datos
                self.puente.emitir(evento, self.t, datos)
            time.sleep(0.005)

    def _programar(self, demora, evento, datos):
        self.pendientes.append((time.monotonic() + demora, evento, datos))

    def _atender(self, cliente, mensaje):
        cmd, args = mensaje["cmd"], mensaje.get("args", {})
        self.ordenes.append((cmd, args))
        responder = lambda ok, r=None, e=None: self.puente.responder(cliente, mensaje.get("id"), ok, r, e)  # noqa: E731
        if cmd == "estado":
            responder(True, {"contexto": {"posicion": list(self.posicion)}, "tarea": self.tarea, "ultima_tarea": self.ultima})
        elif cmd == "ruta":
            meta = {"ENTREGA_2": (3.76, 4.31), "ENTREGA_3": (-3.61, 4.31)}[args["destino"]]
            metros, conocido = self.ruta_fn(self.posicion, meta)
            responder(True, {"destino": args["destino"], "hay_ruta": metros is not None, "metros_por_ruta": metros,
                             "metros_en_linea_recta": 0.0, "mapa_conocido_m2": conocido})
        elif cmd == "destinos":
            responder(True, {"destinos": [{"nombre": "ENTREGA_2", "posicion": [3.76, 4.31], "distancia": 0.76},
                                          {"nombre": "ENTREGA_3", "posicion": [-3.61, 4.31], "distancia": 7.0}]})
        elif cmd == "escanear":
            responder(True, {"lidar": {"front": 0.8}, "lado_sugerido_para_evadir": "LEFT"})
        elif cmd == "parar":
            if self.tarea:
                self.pendientes.clear()
                fin = dict(self.tarea, estado="cancelada", motivo="parar", detalle="parar")
                self._programar(0, "tarea_terminada", fin)
            responder(True, {"detenido": True})
        elif cmd in ("ir_a", "ir_a_punto", "girar", "avanzar"):
            if cmd == "ir_a" and args.get("destino") not in ("ENTREGA_2", "ENTREGA_3", "LENTO"):
                responder(False, e="Destino desconocido")
                return
            self.contador += 1
            tid = self.contador
            self.tarea = {"tarea_id": tid, "tipo": cmd, "args": args, "estado": "en_curso", "inicio": self.t}
            responder(True, {"tarea_id": tid})
            self._programar(0, "tarea_iniciada", dict(self.tarea))
            if cmd == "ir_a" and args["destino"] == "ENTREGA_3":
                self._programar(0.05, "evasion_iniciada", {"tarea_id": tid, "direccion": "RIGHT", "intento": 1})
                self._programar(0.15, "tarea_terminada", dict(self.tarea, estado=self.estado_e3, motivo=self.motivo_e3,
                                                               detalle="sin salida", fin=self.t + 45,
                                                               contexto={"modo": "AVOID_TURN", "posicion": list(self.posicion)}))
            elif cmd == "ir_a_punto":
                self.posicion = [args["x"], args["y"]]
                self._programar(0.1, "tarea_terminada", dict(self.tarea, estado="exito", motivo="llego", detalle="ok", fin=self.t + 10,
                                                             contexto={"posicion": list(self.posicion)}))
            elif cmd == "ir_a" and args["destino"] == "LENTO":
                self._programar(5.0, "tarea_terminada", dict(self.tarea, estado="exito", motivo="llego", detalle="", fin=self.t + 99))
            else:
                motivo = "llego" if cmd == "ir_a" else "maniobra_completa"
                self._programar(0.1, "tarea_terminada", dict(self.tarea, estado="exito", motivo=motivo, detalle="ok", fin=self.t + 10))
        else:
            responder(False, e="Comando desconocido")

    def cerrar(self):
        self.activo = False
        self.hilo.join(timeout=1)
        self.puente.cerrar()


class TestHerramientas(unittest.TestCase):
    def setUp(self):
        self.robot = RobotSimulado()
        herramientas.PUERTO = self.robot.puente.puerto

    def tearDown(self):
        self.robot.cerrar()

    def test_ir_a_exito(self):
        r = herramientas.ir_a("ENTREGA_2")
        self.assertEqual((r["estado"], r["motivo"]), ("exito", "llego"))
        self.assertEqual(r["sugerencia"], "Objetivo cumplido.")

    def test_ir_a_fallo_trae_motivo_eventos_y_sugerencia(self):
        r = herramientas.ir_a("ENTREGA_3")
        self.assertEqual((r["estado"], r["motivo"]), ("fallo", "evasion_sin_salida"))
        self.assertEqual([e["evento"] for e in r["eventos"]], ["evasion_iniciada"])
        self.assertIn("retroceder", r["sugerencia"])
        self.assertEqual(r["duracion_simulada_s"], 45.0)

    def test_destino_invalido_devuelve_error_legible(self):
        r = herramientas.ir_a("Marte")
        self.assertIn("Destino desconocido", r["error"])

    def test_sin_esperar_y_luego_esperar_tarea(self):
        r = herramientas.ir_a("ENTREGA_2", esperar=False)
        self.assertEqual(r["estado"], "en_curso")
        final = herramientas.esperar_tarea(r["tarea_id"], timeout_s=5)
        self.assertEqual(final["motivo"], "llego")

    def test_espera_agotada_deja_tarea_en_curso(self):
        r = herramientas.ir_a("LENTO", timeout_s=0.3)
        self.assertEqual(r["estado"], "en_curso")
        self.assertIn("esperar_tarea", r["detalle"])

    def test_parar_mientras_otra_herramienta_espera(self):
        resultado = {}
        hilo = threading.Thread(target=lambda: resultado.update(herramientas.ir_a("LENTO", timeout_s=10)))
        hilo.start()
        time.sleep(0.3)
        self.assertTrue(herramientas.parar()["detenido"])
        hilo.join(timeout=5)
        self.assertEqual((resultado["estado"], resultado["motivo"]), ("cancelada", "parar"))

    def test_opciones_llegan_al_robot_y_las_vacias_no(self):
        herramientas.ir_a("ENTREGA_2", velocidad=0.15, al_ver_persona="detenerse", distancia_persona=2.0)
        self.assertIn(("ir_a", {"destino": "ENTREGA_2", "velocidad": 0.15, "al_ver_persona": "detenerse",
                                "distancia_persona": 2.0}), self.robot.ordenes)
        herramientas.girar(30)
        self.assertIn(("girar", {"grados": 30}), self.robot.ordenes)

    def test_interrumpir_con_evento_corta_la_espera(self):
        r = herramientas.ir_a("ENTREGA_3", interrumpir_con=["evasion_iniciada"])
        self.assertEqual((r["estado"], r["interrumpida_por"]), ("en_curso", "evasion_iniciada"))
        self.assertEqual(r["evento"]["direccion"], "RIGHT")
        final = herramientas.esperar_tarea(r["tarea_id"], timeout_s=5)
        self.assertEqual(final["motivo"], "evasion_sin_salida")

    def test_persona_cerca_trae_sugerencia(self):
        self.robot.estado_e3, self.robot.motivo_e3 = "detenida", "persona_cerca"
        r = herramientas.ir_a("ENTREGA_3")
        self.assertEqual((r["estado"], r["motivo"]), ("detenida", "persona_cerca"))
        self.assertIn("PRIORIDAD", r["sugerencia"])

    def test_ir_a_punto(self):
        r = herramientas.ir_a_punto(1.5, -2.0)
        self.assertEqual((r["estado"], r["motivo"]), ("exito", "llego"))
        self.assertIn(("ir_a_punto", {"x": 1.5, "y": -2.0}), self.robot.ordenes)

    def test_sugerencia_bloqueado_por_seguridad(self):
        self.assertIn("ir_a_punto", herramientas.SUGERENCIAS["bloqueado_por_seguridad"])

    def test_maniobras(self):
        self.assertEqual(herramientas.girar(90)["motivo"], "maniobra_completa")
        self.assertEqual(herramientas.avanzar(-0.3)["motivo"], "maniobra_completa")
        self.assertIn(("girar", {"grados": 90}), self.robot.ordenes)

    def test_sin_robot_devuelve_error(self):
        herramientas.PUERTO = 1
        self.assertIn("Webots", herramientas.estado_robot()["error"])


class TestServidorMCP(unittest.TestCase):
    def setUp(self):
        self.robot = RobotSimulado()
        herramientas.PUERTO = self.robot.puente.puerto

    def tearDown(self):
        self.robot.cerrar()

    def test_lista_de_herramientas_y_llamada(self):
        from mcp import Client
        from servidor_mcp import servidor

        async def flujo():
            async with Client(servidor) as cliente:
                nombres = sorted(t.name for t in (await cliente.list_tools()).tools)
                resultado = await cliente.call_tool("ir_a", {"destino": "ENTREGA_3"})
                return nombres, resultado

        nombres, resultado = asyncio.run(flujo())
        self.assertEqual(nombres, ["avanzar", "distancia_por_ruta", "escanear_entorno", "esperar_tarea", "estado_robot",
                                   "girar", "ir_a", "ir_a_punto", "listar_destinos", "parar"])

        self.assertFalse(resultado.is_error)
        datos = resultado.structured_content or json.loads(resultado.content[0].text)
        datos = datos.get("result", datos)
        self.assertEqual(datos["motivo"], "evasion_sin_salida")


if __name__ == "__main__":
    unittest.main()
