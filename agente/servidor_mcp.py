"""Servidor MCP del MiR100: expone las herramientas del robot a cualquier cliente MCP
(Claude Code, Claude Desktop o el agente de la parte 3).

Requiere la simulacion corriendo en Webots (el controlador abre el puente en
127.0.0.1:5555, o en MIR_PUENTE_PUERTO).

    python3 agente/servidor_mcp.py          # transporte stdio
"""
import functools
import os
import sys

import anyio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herramientas  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

INSTRUCCIONES = """\
Controlas un robot movil MiR100 en una fabrica simulada (Webots).
Principios:
- Las herramientas de movimiento esperan a que la tarea termine y devuelven estado
  (exito / fallo / cancelada), motivo, detalle, contexto, eventos y una sugerencia.
- La capa de seguridad del robot siempre tiene la ultima palabra: si dice que no se
  puede, no insistas con la misma accion; cambia de estrategia o informa.
- Ante un fallo: escanear_entorno y, en vez de repetir lo mismo, rodea el obstaculo con
  ir_a_punto hacia el lado libre; despues reintenta ir_a. Si tras 3 intentos el robot
  no se acerca al destino, deten el robot y explica con claridad por que no se puede seguir.
- Nunca reanudes una tarea que un operador cancelo desde el teclado sin consultar.
"""

servidor = MCPServer(name="mir100", title="Robot MiR100 (Webots)", instructions=INSTRUCCIONES)


async def _en_hilo(funcion, *args, **kwargs):
    # Las herramientas bloquean (sockets): se ejecutan en un hilo para no frenar el servidor
    # y permitir, por ejemplo, llamar a parar mientras ir_a sigue esperando.
    return await anyio.to_thread.run_sync(functools.partial(funcion, *args, **kwargs))


@servidor.tool()
async def estado_robot() -> dict:
    """Posicion, orientacion, modo, destino actual, lecturas del LiDAR, persona detectada,
    la tarea en curso y la ultima tarea terminada (con su motivo)."""
    return await _en_hilo(herramientas.estado_robot)


@servidor.tool()
async def listar_destinos() -> dict:
    """Destinos de entrega disponibles (ENTREGA_1, ENTREGA_2, ENTREGA_3) con su posicion
    y la distancia en linea recta desde el robot."""
    return await _en_hilo(herramientas.listar_destinos)


@servidor.tool()
async def escanear_entorno() -> dict:
    """Distancias del LiDAR por sector en metros (left, front_left, front, front_right,
    right, back; 0.0 = sin datos, se trata como bloqueado), si el frente esta libre, si
    puede girar a cada lado, el lado sugerido para evadir y si hay una persona."""
    return await _en_hilo(herramientas.escanear_entorno)


@servidor.tool()
async def ir_a(destino: str, esperar: bool = True, timeout_s: float = 300.0) -> dict:
    """Navega a un destino de entrega ("ENTREGA_3", "entrega 3" o "3"). El robot gira hacia
    el destino, avanza y esquiva obstaculos solo. Con esperar=true devuelve el resultado
    final: estado, motivo (llego, sin_espacio_lateral, limite_evasiones, evasion_sin_salida,
    sin_progreso, lidar_sin_datos...), detalle, eventos y sugerencia."""
    return await _en_hilo(herramientas.ir_a, destino, esperar=esperar, timeout_s=timeout_s)


@servidor.tool()
async def ir_a_punto(x: float, y: float, esperar: bool = True, timeout_s: float = 180.0) -> dict:
    """Navega a un punto (x, y) en metros, en las mismas coordenadas del mundo que
    estado_robot y listar_destinos. Sirve como punto intermedio para rodear un obstaculo
    (una mesa, una maquina) antes de seguir con ir_a. Mismo resultado que ir_a."""
    return await _en_hilo(herramientas.ir_a_punto, x, y, esperar=esperar, timeout_s=timeout_s)


@servidor.tool()
async def girar(grados: float, esperar: bool = True) -> dict:
    """Gira en el lugar. Positivo = izquierda, negativo = derecha, maximo 180 en valor
    absoluto. Util para orientarse hacia el lado libre antes de reintentar ir_a. Cancela
    la tarea en curso."""
    return await _en_hilo(herramientas.girar, grados, esperar=esperar)


@servidor.tool()
async def avanzar(metros: float, esperar: bool = True) -> dict:
    """Avanza en linea recta; negativo = retroceder. Maximo 1.0 m en valor absoluto. Util
    para salir de un atasco (retroceder 0.3-0.5 m). Cancela la tarea en curso."""
    return await _en_hilo(herramientas.avanzar, metros, esperar=esperar)


@servidor.tool()
async def parar() -> dict:
    """Detiene el robot y cancela la tarea en curso."""
    return await _en_hilo(herramientas.parar)


@servidor.tool()
async def esperar_tarea(tarea_id: int, timeout_s: float = 120.0) -> dict:
    """Espera el final de una tarea lanzada con esperar=false (o cuya espera se agoto)
    y devuelve su resultado."""
    return await _en_hilo(herramientas.esperar_tarea, tarea_id, timeout_s=timeout_s)


if __name__ == "__main__":
    servidor.run()
