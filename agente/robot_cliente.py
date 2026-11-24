"""Cliente del puente de comandos del MiR100 (ver mir100/controllers/keyboard_control/puente.py).

Uso:
    with RobotCliente() as robot:
        robot.comando("ir_a", destino="ENTREGA_3")
        evento = robot.esperar_evento(lambda e: e["evento"] == "tarea_terminada", timeout=300)
"""
import collections
import itertools
import json
import os
import socket
import time

HOST_POR_DEFECTO = "127.0.0.1"
PUERTO_POR_DEFECTO = int(os.getenv("MIR_PUENTE_PUERTO", "5555"))


class RobotError(Exception):
    """El robot respondio ok=false, o no hay conexion con el puente."""


class RobotCliente:
    def __init__(self, host=HOST_POR_DEFECTO, puerto=PUERTO_POR_DEFECTO, timeout=5.0):
        self.host = host
        self.puerto = puerto
        self.timeout = timeout
        self.socket = None
        self.buffer = b""
        self.eventos = collections.deque()
        self.ids = itertools.count(1)

    def conectar(self):
        try:
            self.socket = socket.create_connection((self.host, self.puerto), timeout=self.timeout)
        except OSError as error:
            raise RobotError("No hay conexion con el robot en {}:{} ({}). Esta Webots corriendo la simulacion?".format(self.host, self.puerto, error))
        return self

    def cerrar(self):
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def __enter__(self):
        return self.conectar()

    def __exit__(self, *exc):
        self.cerrar()

    def comando(self, cmd, timeout=None, **args):
        """Envia un comando y devuelve 'resultado'. Lanza RobotError si ok=false."""
        if self.socket is None:
            self.conectar()
        id_pedido = next(self.ids)
        linea = json.dumps({"id": id_pedido, "cmd": cmd, "args": args}, ensure_ascii=False) + "\n"
        try:
            self.socket.sendall(linea.encode("utf-8"))
        except OSError as error:
            self.cerrar()
            raise RobotError("Se perdio la conexion con el robot: {}".format(error))
        limite = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while True:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RobotError("El robot no respondio a '{}' a tiempo (la simulacion puede estar en pausa)".format(cmd))
            mensaje = self._leer_mensaje(restante)
            if mensaje is None:
                continue
            if mensaje.get("tipo") == "evento":
                self.eventos.append(mensaje)
                continue
            if mensaje.get("id") == id_pedido:
                if mensaje.get("ok"):
                    return mensaje.get("resultado", {})
                raise RobotError(mensaje.get("error", "error desconocido"))

    def esperar_evento(self, predicado=None, timeout=None):
        """Devuelve el primer evento que cumpla el predicado, o None si se agota el tiempo."""
        predicado = predicado or (lambda evento: True)
        for evento in list(self.eventos):
            if predicado(evento):
                self.eventos.remove(evento)
                return evento
        limite = None if timeout is None else time.monotonic() + timeout
        while True:
            restante = None if limite is None else limite - time.monotonic()
            if restante is not None and restante <= 0:
                return None
            mensaje = self._leer_mensaje(restante if restante is not None else 1.0)
            if mensaje is None or mensaje.get("tipo") != "evento":
                continue
            if predicado(mensaje):
                return mensaje
            self.eventos.append(mensaje)

    def tomar_eventos(self, espera=0.0):
        """Lee lo que haya llegado (hasta 'espera' segundos) y devuelve todos los eventos pendientes."""
        limite = time.monotonic() + espera
        while True:
            restante = limite - time.monotonic()
            mensaje = self._leer_mensaje(max(restante, 0.01))
            if mensaje is not None and mensaje.get("tipo") == "evento":
                self.eventos.append(mensaje)
            if mensaje is None and restante <= 0:
                break
        eventos = list(self.eventos)
        self.eventos.clear()
        return eventos

    def _leer_mensaje(self, timeout):
        if self.socket is None:
            self.conectar()
        while b"\n" not in self.buffer:
            self.socket.settimeout(max(timeout, 0.01))
            try:
                datos = self.socket.recv(65536)
            except socket.timeout:
                return None
            except OSError as error:
                self.cerrar()
                raise RobotError("Se perdio la conexion con el robot: {}".format(error))
            if not datos:
                self.cerrar()
                raise RobotError("El robot cerro la conexion (se detuvo la simulacion?)")
            self.buffer += datos
        linea, self.buffer = self.buffer.split(b"\n", 1)
        try:
            return json.loads(linea.decode("utf-8"))
        except ValueError:
            return None
