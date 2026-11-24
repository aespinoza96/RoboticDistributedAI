"""Tests del puente de comandos y su cliente, sin Webots.

    python3 -m unittest discover -s tests -v
"""
import os
import socket
import sys
import threading
import time
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "mir100", "controllers", "keyboard_control"))
sys.path.insert(0, os.path.join(RAIZ, "agente"))

from puente import PuenteComandos  # noqa: E402
from robot_cliente import RobotCliente, RobotError  # noqa: E402


class RobotFalso:
    """Simula el bucle del controlador: poll() en cada 'paso' y respuestas inmediatas."""

    def __init__(self):
        self.puente = PuenteComandos(puerto=0)
        self.recibidos = []
        self.activo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.activo:
            for cliente, mensaje in self.puente.poll():
                self.recibidos.append(mensaje)
                if mensaje["cmd"] == "eco":
                    self.puente.responder(cliente, mensaje.get("id"), True, {"args": mensaje.get("args", {})})
                elif mensaje["cmd"] == "evento_y_respuesta":
                    self.puente.emitir("tarea_iniciada", 1.0, {"tarea_id": 7})
                    self.puente.responder(cliente, mensaje.get("id"), True, {"tarea_id": 7})
                else:
                    self.puente.responder(cliente, mensaje.get("id"), False, error="Comando desconocido")
            time.sleep(0.005)

    def cerrar(self):
        self.activo = False
        self.hilo.join(timeout=1)
        self.puente.cerrar()


class TestPuente(unittest.TestCase):
    def setUp(self):
        self.robot = RobotFalso()
        self.cliente = RobotCliente(puerto=self.robot.puente.puerto, timeout=2.0).conectar()

    def tearDown(self):
        self.cliente.cerrar()
        self.robot.cerrar()

    def test_comando_y_respuesta(self):
        resultado = self.cliente.comando("eco", destino="ENTREGA_3", n=2)
        self.assertEqual(resultado, {"args": {"destino": "ENTREGA_3", "n": 2}})

    def test_error_del_robot_lanza_excepcion(self):
        with self.assertRaises(RobotError) as contexto:
            self.cliente.comando("volar")
        self.assertIn("desconocido", str(contexto.exception))

    def test_evento_antes_de_la_respuesta_queda_guardado(self):
        resultado = self.cliente.comando("evento_y_respuesta")
        self.assertEqual(resultado["tarea_id"], 7)
        evento = self.cliente.esperar_evento(lambda e: e["evento"] == "tarea_iniciada", timeout=1)
        self.assertIsNotNone(evento)
        self.assertEqual(evento["datos"]["tarea_id"], 7)

    def test_evento_espontaneo_llega_a_todos_los_clientes(self):
        otro = RobotCliente(puerto=self.robot.puente.puerto, timeout=2.0).conectar()
        try:
            self.cliente.comando("eco")
            otro.comando("eco")  # asegura que ambos clientes ya fueron aceptados
            self.robot.puente.emitir("tarea_terminada", 5.0, {"estado": "fallo", "motivo": "sin_progreso"})
            for c in (self.cliente, otro):
                evento = c.esperar_evento(timeout=1)
                self.assertEqual(evento["datos"]["motivo"], "sin_progreso")
        finally:
            otro.cerrar()

    def test_json_invalido_responde_error_sin_cortar(self):
        crudo = socket.create_connection(("127.0.0.1", self.robot.puente.puerto), timeout=2)
        try:
            crudo.sendall(b"esto no es json\n")
            respuesta = crudo.recv(4096).decode("utf-8")
            self.assertIn('"ok": false', respuesta)
            self.assertIn("JSON invalido", respuesta)
        finally:
            crudo.close()
        self.assertEqual(self.cliente.comando("eco", x=1), {"args": {"x": 1}})

    def test_esperar_evento_con_timeout_devuelve_none(self):
        self.assertIsNone(self.cliente.esperar_evento(timeout=0.2))

    def test_sin_robot_da_error_claro(self):
        libre = socket.socket()
        libre.bind(("127.0.0.1", 0))
        puerto = libre.getsockname()[1]
        libre.close()
        with self.assertRaises(RobotError) as contexto:
            RobotCliente(puerto=puerto, timeout=0.5).conectar()
        self.assertIn("Webots", str(contexto.exception))


class TestNormalizarDestino(unittest.TestCase):
    def test_variantes(self):
        # Copia de la funcion del controlador (no se puede importar sin Webots).
        import ast
        ruta = os.path.join(RAIZ, "mir100", "controllers", "keyboard_control", "keyboard_control.py")
        with open(ruta, encoding="utf-8") as archivo:
            arbol = ast.parse(archivo.read())
        nodo = next(n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == "normalize_destination")
        espacio = {}
        exec(compile(ast.Module([nodo], []), ruta, "exec"), espacio)
        normalizar = espacio["normalize_destination"]
        nombres = {"ENTREGA_1": 0, "ENTREGA_2": 0, "ENTREGA_3": 0}
        for entrada in ("ENTREGA_3", "entrega 3", "Entrega-3", "3", 3):
            self.assertEqual(normalizar(entrada, nombres), "ENTREGA_3")
        self.assertIsNone(normalizar("cocina", nombres))
        self.assertIsNone(normalizar("4", nombres))


if __name__ == "__main__":
    unittest.main()
