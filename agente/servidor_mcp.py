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
- PRIORIDAD: si el robot ve una persona a 0.6 m o menos se detiene solo y la tarea termina
  con estado "detenida" y motivo "persona_cerca" (con su posicion). Informalo siempre.
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
async def distancia_por_ruta(destino: str) -> dict:
    """Cuantos metros de ruta faltan hasta un destino segun el mapa que el robot fue
    armando (A*, rodeando lo conocido), la distancia en linea recta y cuanto mapa conoce
    (m2). hay_ruta=false si con lo conocido no hay camino. No mueve el robot."""
    return await _en_hilo(herramientas.distancia_por_ruta, destino)


@servidor.tool()
async def ir_a(destino: str, esperar: bool = True, timeout_s: float = 300.0, velocidad: float | None = None,
               al_ver_persona: str | None = None, distancia_persona: float | None = None,
               interrumpir_con: list[str] | None = None) -> dict:
    """Navega a un destino de entrega ("ENTREGA_3", "entrega 3" o "3") planificando la ruta
    sobre el mapa que el robot va armando. Con esperar=true devuelve el resultado final:
    estado (exito / fallo / cancelada / detenida), motivo, detalle, contexto, eventos y
    sugerencia. velocidad: m/s maxima (0.05-0.40, defecto 0.30).
    Opciones (todas opcionales):
    - al_ver_persona: "rodear" (por defecto: va despacio y la rodea), "detenerse" (la
      orden TERMINA en cuanto ve una persona a distancia_persona o menos: estado
      "detenida", motivo "persona_a_la_vista") o "esperar" (frena mientras haya una persona
      delante a distancia_persona o menos y sigue cuando se aparta).
    - distancia_persona: metros de holgura para "detenerse"/"esperar" (defecto 3).
    - interrumpir_con: eventos que cortan la ESPERA (no la tarea) para decidir en el medio,
      p. ej. ["persona_detectada", "obstaculo_alto"]: devuelve estado "en_curso" con
      interrumpida_por; despues parar, esperar_tarea u otra orden."""
    return await _en_hilo(herramientas.ir_a, destino, esperar=esperar, timeout_s=timeout_s, velocidad=velocidad,
                          al_ver_persona=al_ver_persona, distancia_persona=distancia_persona, interrumpir_con=interrumpir_con)


@servidor.tool()
async def ir_a_punto(x: float, y: float, esperar: bool = True, timeout_s: float = 180.0, velocidad: float | None = None,
                     al_ver_persona: str | None = None, distancia_persona: float | None = None,
                     interrumpir_con: list[str] | None = None) -> dict:
    """Navega a un punto (x, y) en metros, en las coordenadas del mundo de estado_robot y
    listar_destinos. Sirve como punto intermedio para rodear o para explorar. Mismo
    resultado y opciones que ir_a.
    Opciones (todas opcionales):
    - al_ver_persona: "rodear" (por defecto: va despacio y la rodea), "detenerse" (la
      orden TERMINA en cuanto ve una persona a distancia_persona o menos: estado
      "detenida", motivo "persona_a_la_vista") o "esperar" (frena mientras haya una persona
      delante a distancia_persona o menos y sigue cuando se aparta).
    - distancia_persona: metros de holgura para "detenerse"/"esperar" (defecto 3).
    - interrumpir_con: eventos que cortan la ESPERA (no la tarea) para decidir en el medio,
      p. ej. ["persona_detectada", "obstaculo_alto"]: devuelve estado "en_curso" con
      interrumpida_por; despues parar, esperar_tarea u otra orden."""
    return await _en_hilo(herramientas.ir_a_punto, x, y, esperar=esperar, timeout_s=timeout_s, velocidad=velocidad,
                          al_ver_persona=al_ver_persona, distancia_persona=distancia_persona, interrumpir_con=interrumpir_con)


@servidor.tool()
async def girar(grados: float, esperar: bool = True, al_ver_persona: str | None = None,
                distancia_persona: float | None = None, interrumpir_con: list[str] | None = None) -> dict:
    """Gira en el lugar. Positivo = izquierda, negativo = derecha, maximo 180 en valor
    absoluto. Para BUSCAR a alguien: girar con al_ver_persona="detenerse" para que pare
    mirando a la primera persona que vea. Cancela la tarea en curso.
    Opciones (todas opcionales):
    - al_ver_persona: "rodear" (por defecto: va despacio y la rodea), "detenerse" (la
      orden TERMINA en cuanto ve una persona a distancia_persona o menos: estado
      "detenida", motivo "persona_a_la_vista") o "esperar" (frena mientras haya una persona
      delante a distancia_persona o menos y sigue cuando se aparta).
    - distancia_persona: metros de holgura para "detenerse"/"esperar" (defecto 3).
    - interrumpir_con: eventos que cortan la ESPERA (no la tarea) para decidir en el medio,
      p. ej. ["persona_detectada", "obstaculo_alto"]: devuelve estado "en_curso" con
      interrumpida_por; despues parar, esperar_tarea u otra orden."""
    return await _en_hilo(herramientas.girar, grados, esperar=esperar, al_ver_persona=al_ver_persona,
                          distancia_persona=distancia_persona, interrumpir_con=interrumpir_con)


@servidor.tool()
async def avanzar(metros: float, esperar: bool = True, al_ver_persona: str | None = None,
                  distancia_persona: float | None = None, interrumpir_con: list[str] | None = None) -> dict:
    """Avanza en linea recta; negativo = retroceder. Maximo 1.0 m en valor absoluto. Util
    para salir de un atasco. Cancela la tarea en curso.
    Opciones (todas opcionales):
    - al_ver_persona: "rodear" (por defecto: va despacio y la rodea), "detenerse" (la
      orden TERMINA en cuanto ve una persona a distancia_persona o menos: estado
      "detenida", motivo "persona_a_la_vista") o "esperar" (frena mientras haya una persona
      delante a distancia_persona o menos y sigue cuando se aparta).
    - distancia_persona: metros de holgura para "detenerse"/"esperar" (defecto 3).
    - interrumpir_con: eventos que cortan la ESPERA (no la tarea) para decidir en el medio,
      p. ej. ["persona_detectada", "obstaculo_alto"]: devuelve estado "en_curso" con
      interrumpida_por; despues parar, esperar_tarea u otra orden."""
    return await _en_hilo(herramientas.avanzar, metros, esperar=esperar, al_ver_persona=al_ver_persona,
                          distancia_persona=distancia_persona, interrumpir_con=interrumpir_con)


@servidor.tool()
async def parar() -> dict:
    """Detiene el robot y cancela la tarea en curso."""
    return await _en_hilo(herramientas.parar)


@servidor.tool()
async def esperar_tarea(tarea_id: int, timeout_s: float = 120.0, interrumpir_con: list[str] | None = None) -> dict:
    """Espera el final de una tarea lanzada con esperar=false (o cuya espera se agoto)
    y devuelve su resultado."""
    return await _en_hilo(herramientas.esperar_tarea, tarea_id, timeout_s=timeout_s, interrumpir_con=interrumpir_con)


if __name__ == "__main__":
    servidor.run()
