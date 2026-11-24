"""Puente de comandos del MiR100: servidor TCP local, no bloqueante.

Protocolo: una linea JSON por mensaje (UTF-8, terminada en "\\n").

Pedido del cliente:
    {"id": 1, "cmd": "ir_a", "args": {"destino": "ENTREGA_3"}}
Respuesta del robot (mismo id):
    {"tipo": "respuesta", "id": 1, "ok": true, "resultado": {...}}
    {"tipo": "respuesta", "id": 1, "ok": false, "error": "..."}
Evento espontaneo del robot (a todos los clientes):
    {"tipo": "evento", "evento": "tarea_terminada", "t": 12.3, "datos": {...}}

El controlador llama a poll() una vez por paso de simulacion. Nada aqui
bloquea robot.step() ni toca la API de Webots.
"""
import json
import socket

PUERTO_POR_DEFECTO = 5555
MAX_LINEA = 64 * 1024


class _Cliente:
    def __init__(self, conexion, direccion):
        self.conexion = conexion
        self.direccion = direccion
        self.buffer = b""


class PuenteComandos:
    def __init__(self, host="127.0.0.1", puerto=PUERTO_POR_DEFECTO):
        self.clientes = []
        self.servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.servidor.bind((host, puerto))
        self.servidor.listen(4)
        self.servidor.setblocking(False)
        self.puerto = self.servidor.getsockname()[1]

    def poll(self):
        """Acepta conexiones nuevas y devuelve [(cliente, mensaje)] completos."""
        self._aceptar()
        mensajes = []
        for cliente in list(self.clientes):
            try:
                datos = cliente.conexion.recv(MAX_LINEA)
            except BlockingIOError:
                continue
            except OSError:
                self._quitar(cliente)
                continue
            if not datos:
                self._quitar(cliente)
                continue
            cliente.buffer += datos
            if len(cliente.buffer) > MAX_LINEA and b"\n" not in cliente.buffer:
                self._enviar(cliente, {"tipo": "respuesta", "id": None, "ok": False, "error": "linea demasiado larga"})
                cliente.buffer = b""
                continue
            while b"\n" in cliente.buffer:
                linea, cliente.buffer = cliente.buffer.split(b"\n", 1)
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    mensaje = json.loads(linea.decode("utf-8"))
                    if not isinstance(mensaje, dict) or not isinstance(mensaje.get("cmd"), str):
                        raise ValueError("falta 'cmd'")
                except (ValueError, UnicodeDecodeError) as error:
                    self._enviar(cliente, {"tipo": "respuesta", "id": None, "ok": False, "error": "JSON invalido: {}".format(error)})
                    continue
                if not isinstance(mensaje.get("args", {}), dict):
                    self._enviar(cliente, {"tipo": "respuesta", "id": mensaje.get("id"), "ok": False, "error": "'args' debe ser un objeto"})
                    continue
                mensajes.append((cliente, mensaje))
        return mensajes

    def responder(self, cliente, id_pedido, ok, resultado=None, error=None):
        respuesta = {"tipo": "respuesta", "id": id_pedido, "ok": bool(ok)}
        if ok:
            respuesta["resultado"] = resultado if resultado is not None else {}
        else:
            respuesta["error"] = error or "error desconocido"
        self._enviar(cliente, respuesta)

    def emitir(self, evento, t, datos=None):
        mensaje = {"tipo": "evento", "evento": evento, "t": round(float(t), 2), "datos": datos or {}}
        for cliente in list(self.clientes):
            self._enviar(cliente, mensaje)

    def cerrar(self):
        for cliente in list(self.clientes):
            self._quitar(cliente)
        try:
            self.servidor.close()
        except OSError:
            pass

    def _aceptar(self):
        while True:
            try:
                conexion, direccion = self.servidor.accept()
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            conexion.setblocking(False)
            self.clientes.append(_Cliente(conexion, direccion))

    def _enviar(self, cliente, mensaje):
        datos = (json.dumps(mensaje, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            # Mensajes pequenos: un envio bloqueante corto es aceptable y evita
            # perder datos con el socket no bloqueante.
            cliente.conexion.setblocking(True)
            cliente.conexion.settimeout(0.5)
            cliente.conexion.sendall(datos)
        except OSError:
            self._quitar(cliente)
            return
        finally:
            try:
                cliente.conexion.setblocking(False)
            except OSError:
                pass

    def _quitar(self, cliente):
        if cliente in self.clientes:
            self.clientes.remove(cliente)
        try:
            cliente.conexion.close()
        except OSError:
            pass
