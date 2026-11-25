"""Herramientas del MiR100 para un agente: funciones simples sobre el puente.

Las usan el servidor MCP (servidor_mcp.py) y, en la parte 3, el agente autonomo.
Cada llamada abre su propia conexion con el robot, asi una espera larga
(ir_a) no impide llamar a parar() desde otra herramienta.

Todas devuelven un dict serializable a JSON. Los errores de conexion o de
argumentos se devuelven como {"error": "..."} en lugar de lanzar excepciones,
para que el agente pueda leerlos y decidir.
"""
import time

from robot_cliente import PUERTO_POR_DEFECTO, RobotCliente, RobotError

ESPERA_MAX_SEGUNDOS = 600.0
# Eventos que SIEMPRE cortan la espera de una tool, para que el agente se entere al momento.
INTERRUPCIONES_SIEMPRE = {"persona_cerca"}

# Que puede intentar el agente ante cada motivo de fallo. No es una orden: el
# agente decide, y la capa de seguridad del robot sigue teniendo la ultima palabra.
SUGERENCIAS = {
    "llego": "Objetivo cumplido.",
    "maniobra_completa": "Maniobra cumplida.",
    "sin_espacio_lateral": "Obstaculo al frente sin espacio para girar. Prueba escanear_entorno, retroceder 0.3-0.5 m con avanzar(metros negativos) y volver a intentar ir_a.",
    "limite_evasiones": "El robot esquivo varias veces sin llegar. Prueba retroceder, girar hacia el lado con mas espacio (escanear_entorno) y reintentar ir_a; si se repite, la ruta probablemente esta bloqueada.",
    "evasion_sin_salida": "La evasion no encontro salida. Prueba retroceder 0.5 m, girar 45-90 grados hacia el lado con mas espacio y reintentar ir_a.",
    "bloqueado_por_seguridad": "La seguridad freno la navegacion (ver detalle: que regla y en que modo). Mira obstaculos_altos en escanear_entorno y rodea el obstaculo con ir_a_punto hacia un lado libre, en vez de reintentar ir_a igual.",
    "persona_cerca": "PRIORIDAD: el robot vio una persona cerca, se detuvo y le pidio paso en voz alta; retoma la tarea solo en cuanto la persona se aparta (maximo 60 s). Informa al operador. Usa esperar_tarea para seguir la tarea, o parar si la orden lo pide.",
    "persona_sigue_cerca": "Tras la espera de 60 s la persona sigue a 1 m o menos y el robot quedo detenido. Informa al operador; para seguir, rodea a la persona (girar o retroceder y luego ir_a / ir_a_punto) o espera.",
    "persona_a_la_vista": "El robot se detuvo al ver a una persona, como se pidio (ver detalle: distancia, angulo y posicion). Sigue la orden segun lo que pidio el operador.",
    "sin_ruta": "El mapa no tiene camino libre hacia el destino con lo que conoce. Explora con ir_a_punto hacia zonas que el robot aun no vio (por ejemplo por el otro extremo del obstaculo) y vuelve a intentar; si todo esta cerrado, reportalo.",
    "meta_inalcanzable": "El destino cae dentro de un obstaculo segun el mapa; el robot quedo lo mas cerca posible. Informa que hay algo ocupando el punto de entrega.",
    "sin_progreso": "El robot no se desplazo. Revisa escanear_entorno: puede estar encajado; prueba retroceder o girar antes de reintentar.",
    "lidar_sin_datos": "El LiDAR no da lecturas suficientes. No es seguro moverse; informa el problema en lugar de insistir.",
    "maniobra_bloqueada": "La seguridad impidio la maniobra (ver detalle). Elige otra maniobra hacia el lado libre o informa que el robot esta encajado.",
    "tiempo_agotado": "La maniobra no termino a tiempo. Revisa estado_robot y escanear_entorno antes de seguir.",
    "parar": "Tarea cancelada por parar().",
    "reemplazada": "Otra orden reemplazo esta tarea.",
    "teclado": "Un operador cancelo la tarea desde el teclado de Webots; no la reanudes sin consultar.",
}


PUERTO = PUERTO_POR_DEFECTO  # los tests lo cambian para apuntar a un robot falso


def _cliente(puerto):
    return RobotCliente(puerto=puerto or PUERTO, timeout=10.0)


def _llamar(puerto, cmd, **args):
    try:
        with _cliente(puerto) as robot:
            return robot.comando(cmd, **args)
    except RobotError as error:
        return {"error": str(error)}


def _resumen_evento(evento):
    datos = evento.get("datos", {})
    claves = ("estado", "motivo", "detalle", "direccion", "intento", "max_intentos", "segundos", "confianza", "lidar")
    return {"t": evento.get("t"), "evento": evento.get("evento"), **{k: datos[k] for k in claves if k in datos}}


def _esperar(robot, tarea_id, timeout, al_evento=None, interrumpir_con=None):
    """Espera tarea_terminada de tarea_id, guardando los eventos intermedios.

    interrumpir_con = nombres de eventos (p. ej. ["persona_detectada", "obstaculo_alto"]):
    si llega uno, devuelve en ese momento con estado "en_curso" (la tarea SIGUE) para que el
    agente decida: parar, seguir esperando (esperar_tarea) o dar otra orden.
    """
    interrumpir_con = set(interrumpir_con or ()) | INTERRUPCIONES_SIEMPRE
    timeout = min(float(timeout), ESPERA_MAX_SEGUNDOS)
    inicio = time.monotonic()
    linea_de_tiempo = []
    while True:
        restante = timeout - (time.monotonic() - inicio)
        if restante <= 0:
            return {"tarea_id": tarea_id, "estado": "en_curso",
                    "detalle": "La tarea sigue en curso tras {:.0f} s de espera. Usa esperar_tarea o parar.".format(timeout),
                    "eventos": linea_de_tiempo}
        evento = robot.esperar_evento(timeout=min(restante, 5.0))
        if evento is None:
            continue
        datos = evento.get("datos", {})
        if datos.get("tarea_id") not in (None, tarea_id):
            continue
        if evento["evento"] == "tarea_iniciada":
            continue
        if evento["evento"] == "tarea_terminada":
            motivo = datos.get("motivo")
            return {"tarea_id": tarea_id, "tipo": datos.get("tipo"), "args": datos.get("args"),
                    "estado": datos.get("estado"), "motivo": motivo, "detalle": datos.get("detalle"),
                    "duracion_simulada_s": round((datos.get("fin") or 0) - (datos.get("inicio") or 0), 1),
                    "contexto": datos.get("contexto"), "eventos": linea_de_tiempo,
                    "sugerencia": SUGERENCIAS.get(motivo, "")}
        resumen = _resumen_evento(evento)
        linea_de_tiempo.append(resumen)
        if al_evento is not None:
            al_evento(resumen)
        if evento["evento"] in interrumpir_con:
            return {"tarea_id": tarea_id, "estado": "en_curso", "interrumpida_por": evento["evento"],
                    "evento": {k: v for k, v in datos.items() if k != "tarea_id"}, "eventos": linea_de_tiempo,
                    "detalle": datos.get("detalle") or "La tarea SIGUE en curso.",
                    "sugerencia": SUGERENCIAS.get(evento["evento"],
                                                  "La tarea SIGUE en curso. Decide: parar, esperar_tarea para seguir, u otra orden (que la reemplaza).")}


def _orden(puerto, cmd, esperar, timeout, al_evento, interrumpir_con=None, **args):
    args = {k: v for k, v in args.items() if v is not None}
    try:
        with _cliente(puerto) as robot:
            # La conexion se abre antes de la orden para no perder ningun evento.
            resultado = robot.comando(cmd, **args)
            if not esperar:
                return {**resultado, "estado": "en_curso", "detalle": "Orden aceptada. Usa esperar_tarea para conocer el resultado."}
            return _esperar(robot, resultado["tarea_id"], timeout, al_evento, interrumpir_con)
    except RobotError as error:
        return {"error": str(error)}


def estado_robot(puerto=None):
    return _llamar(puerto, "estado")


def listar_destinos(puerto=None):
    return _llamar(puerto, "destinos")


def escanear_entorno(puerto=None):
    return _llamar(puerto, "escanear")


def distancia_por_ruta(destino, puerto=None):
    return _llamar(puerto, "ruta", destino=destino)


def ir_a(destino, esperar=True, timeout_s=300.0, velocidad=None, al_ver_persona=None, distancia_persona=None,
         interrumpir_con=None, puerto=None, al_evento=None):
    return _orden(puerto, "ir_a", esperar, timeout_s, al_evento, interrumpir_con, destino=destino,
                  velocidad=velocidad, al_ver_persona=al_ver_persona, distancia_persona=distancia_persona)


def ir_a_punto(x, y, esperar=True, timeout_s=180.0, velocidad=None, al_ver_persona=None, distancia_persona=None,
               interrumpir_con=None, puerto=None, al_evento=None):
    return _orden(puerto, "ir_a_punto", esperar, timeout_s, al_evento, interrumpir_con, x=x, y=y,
                  velocidad=velocidad, al_ver_persona=al_ver_persona, distancia_persona=distancia_persona)


def girar(grados, esperar=True, timeout_s=60.0, al_ver_persona=None, distancia_persona=None,
          interrumpir_con=None, puerto=None, al_evento=None):
    return _orden(puerto, "girar", esperar, timeout_s, al_evento, interrumpir_con, grados=grados,
                  al_ver_persona=al_ver_persona, distancia_persona=distancia_persona)


def avanzar(metros, esperar=True, timeout_s=60.0, al_ver_persona=None, distancia_persona=None,
            interrumpir_con=None, puerto=None, al_evento=None):
    return _orden(puerto, "avanzar", esperar, timeout_s, al_evento, interrumpir_con, metros=metros,
                  al_ver_persona=al_ver_persona, distancia_persona=distancia_persona)


def parar(puerto=None):
    return _llamar(puerto, "parar")


def esperar_tarea(tarea_id, timeout_s=120.0, interrumpir_con=None, puerto=None, al_evento=None):
    try:
        with _cliente(puerto) as robot:
            estado = robot.comando("estado")
            ultima = estado.get("ultima_tarea") or {}
            actual = estado.get("tarea") or {}
            if actual.get("tarea_id") != tarea_id:
                if ultima.get("tarea_id") == tarea_id:
                    motivo = ultima.get("motivo")
                    return {"tarea_id": tarea_id, "tipo": ultima.get("tipo"), "args": ultima.get("args"),
                            "estado": ultima.get("estado"), "motivo": motivo, "detalle": ultima.get("detalle"),
                            "contexto": ultima.get("contexto"), "eventos": [], "sugerencia": SUGERENCIAS.get(motivo, "")}
                return {"error": "La tarea {} no esta en curso ni es la ultima terminada.".format(tarea_id)}
            return _esperar(robot, tarea_id, timeout_s, al_evento, interrumpir_con)
    except RobotError as error:
        return {"error": str(error)}
