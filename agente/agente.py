"""Agente autonomo del MiR100: recibe una orden en lenguaje natural y la cumple
usando las herramientas del servidor MCP del robot, hasta lograrla o hasta poder
explicar por que no se puede.

    .venv/bin/python agente/agente.py "Vaya al punto de entrega 3"

Requisitos: la simulacion corriendo en Webots (puente en 127.0.0.1:5555 o
MIR_PUENTE_PUERTO) y ANTHROPIC_API_KEY en el entorno o en el .env de la raiz.

Arquitectura:
    orden -> Claude (tool runner) -> herramientas MCP (servidor_mcp.py) -> puente -> robot
El modelo decide los pasos; este programa pone los limites duros (pasos, tiempo,
maniobras) y garantiza siempre un reporte final con el motivo.
"""
import argparse
import asyncio
import datetime
import json
import os
import sys
import time

import anthropic
from anthropic import beta_async_tool
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession, StdioServerParameters, stdio_client

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(DIRECTORIO)

MODELO = "claude-opus-5-5"
HERRAMIENTAS_DE_MOVIMIENTO = {"ir_a", "ir_a_punto", "girar", "avanzar"}
HERRAMIENTAS_DE_NAVEGACION = {"ir_a", "ir_a_punto", "esperar_tarea"}
MEJORA_MINIMA_M = 0.3  # acortar al menos esto la ruta al destino cuenta como progreso
DESCUBRIMIENTO_MINIMO_M2 = 2.0  # descubrir al menos este mapa nuevo tambien cuenta

SISTEMA = """\
Eres el agente de mision de un robot movil MiR100 en una fabrica simulada (Webots).
Recibes una orden de un operador y debes cumplirla de forma autonoma usando las
herramientas del robot, o explicar con claridad por que no se puede.

Como trabajar (decide rapido: una herramienta por paso, sin consultas redundantes):
1. Mira el estado (estado_robot, listar_destinos) y lanza ir_a hacia el destino.
2. Las herramientas de movimiento esperan a que la tarea termine y devuelven estado
   (exito / fallo / cancelada), motivo, detalle, contexto (posicion, orientacion,
   holguras por sector y obstaculos_altos) y una sugerencia.
3. Si falla, planifica un rodeo con lo que sabes: tu posicion, la del destino, el lado
   con mas espacio y los obstaculos_altos (mesas, cintas: el LiDAR no los ve, la camara
   de profundidad si). Usa ir_a_punto(x, y) hacia un punto libre a un costado del
   obstaculo (1-2 m) y despues vuelve a ir_a. Los puntos intermedios son la forma
   principal de rodear; las maniobras sueltas solo para salir de un encierro corto.
4. Si haces una maniobra, que sea una sola con el valor total (un giro de 120 grados,
   no tres de 40). No repitas exactamente lo que ya fallo.
5. Verifica el resultado con estado_robot antes de darlo por cumplido.

Limites:
- La capa de seguridad del robot siempre tiene la ultima palabra. Si una maniobra queda
  bloqueada por seguridad, no insistas en la misma direccion.
- El robot arma un mapa mientras se mueve e ir_a planifica la ruta sobre el. Mientras
  la RUTA que falta se acorte o el robot descubra mapa nuevo, sigue: alejarse en linea
  recta para rodear un obstaculo esta bien. Cada resultado trae una linea de "Progreso"
  (ruta restante, mapa conocido, intentos sin avance); usa distancia_por_ruta para
  comparar opciones. Si no hay ruta (sin_ruta), no es un fracaso: explora con ir_a_punto
  hacia zonas que el robot aun no vio (a 2-4 m, hacia el lado libre) y vuelve a probar.
  Solo al llegar al limite de intentos sin avance, deten el robot con parar y reporta el
  fallo con lo que aprendiste del entorno.
- PRIORIDAD de seguridad: si el robot ve una persona a 1 m o menos, se DETIENE solo y espera
  60 s; la herramienta en curso vuelve en ese momento con estado "en_curso" e
  interrumpida_por "persona_cerca" (distancia, angulo y posicion de la persona). Eso no es
  un fallo: anota que el robot se detuvo por una persona y dilo en el reporte. Despues:
  * si la orden pedia detenerse ante personas, esta cumplida: llama a parar y reporta;
  * si no, llama a esperar_tarea con el mismo tarea_id: el robot le pide paso en voz alta
    y retoma la tarea solo en cuanto la persona se aparta (evento persona_se_aparto); si
    a los 60 s sigue ahi, la tarea termina "detenida" con motivo "persona_sigue_cerca" y
    decides tu (rodearla o reportar). Una orden nueva
    durante la espera tambien espera; nunca avances hacia la persona.
- Ordenes con condiciones: las herramientas de movimiento aceptan opciones.
  * Personas: al_ver_persona="detenerse" (la orden termina sola al ver a alguien, estado
    "detenida"; con distancia_persona eliges a que distancia), "esperar" (frena mientras
    haya alguien delante y sigue despues), "rodear" (la ruta evita a la persona) o
    "continuar" (por defecto: mantiene la ruta y va despacio cerca). Usalas siempre que la
    orden diga que hacer ante personas. Para BUSCAR a alguien: girar en pasos de 90-120
    grados con al_ver_persona="detenerse", y explorar con ir_a_punto si no aparece.
  * Velocidad: velocidad=0.1-0.2 m/s si la orden pide ir con cuidado o despacio.
  * interrumpir_con=["persona_detectada", "obstaculo_alto", ...] corta la espera (la
    tarea sigue) para que decidas en el medio cuando la condicion no encaja en las opciones.
  La posicion de una persona viene en contexto.personas; si su fuente es
  "tamano_aproximado", la distancia es solo una estimacion: dilo asi en el reporte.
- Si una persona bloquea el camino, espera (esperar_tarea) en lugar de esquivarla a la
  fuerza; si sigue bloqueando mucho tiempo, reportalo.
- Si un operador cancelo la tarea desde el teclado, no la reanudes: reporta.
- Si el LiDAR no da datos, no te muevas: reporta.

Al terminar, llama exactamente una vez a reportar_resultado con el resultado, un
resumen para el operador (en espanol, claro y breve), el motivo tecnico si fallo, los
pasos realizados y, si fallo, que haria falta para lograrlo. No llames a ninguna
herramienta despues de reportar.
"""


def cargar_env():
    """Carga el .env de la raiz sin pisar variables ya definidas."""
    ruta = os.path.join(RAIZ, ".env")
    if not os.path.isfile(ruta):
        return
    with open(ruta, encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            nombre, valor = linea.split("=", 1)
            nombre = nombre.strip().removeprefix("export ").strip()
            valor = valor.strip().strip('"').strip("'")
            if nombre and valor and nombre not in os.environ:
                os.environ[nombre] = valor


class Mision:
    """Estado de una ejecucion: reporte final, contadores y bitacora."""

    def __init__(self, orden, max_movimientos, limite_segundos, max_sin_progreso=3):
        self.orden = orden
        self.max_movimientos = max_movimientos
        self.max_sin_progreso = max_sin_progreso
        self.destinos = {}            # nombre -> (x, y), de listar_destinos
        self.objetivo = None          # destino del ultimo ir_a
        self.mejor_distancia = None
        self.intentos_sin_progreso = 0
        self.ultimo_conocido = None
        self.limite = time.monotonic() + limite_segundos
        self.limite_segundos = limite_segundos
        self.inicio = time.monotonic()
        self.movimientos = 0
        self.reporte = None
        self.bitacora = []
        self.ultimo_resultado_robot = None

    def fijar_objetivo(self, herramienta, datos):
        """El destino del ultimo ir_a es el objetivo de la mision."""
        destino = (datos.get("args") or {}).get("destino") if herramienta == "ir_a" else None
        if destino in self.destinos and destino != self.objetivo:
            self.objetivo = destino
            self.mejor_distancia, self.intentos_sin_progreso, self.ultimo_conocido = None, 0, None

    def actualizar_progreso(self, herramienta, datos, ruta=None):
        """Devuelve una linea de progreso para el modelo, o None si no aplica.

        ruta = respuesta de distancia_por_ruta (metros por ruta sobre el mapa y m2 conocidos).
        Progreso = la RUTA que falta se acorta (rodear un obstaculo puede alejar en linea
        recta y aun asi acercar por ruta) o el robot descubrio mapa nuevo (aprender por
        donde no se puede pasar tambien sirve). Sin datos de ruta, se usa la linea recta.
        Un sin_ruta no cuenta como intento: no mueve el robot y la salida es explorar.
        """
        self.fijar_objetivo(herramienta, datos)
        posicion = (datos.get("contexto") or {}).get("posicion")
        if self.objetivo is None or not posicion:
            return None
        gx, gy = self.destinos[self.objetivo]
        recta = ((posicion[0] - gx) ** 2 + (posicion[1] - gy) ** 2) ** 0.5
        ruta = ruta if isinstance(ruta, dict) and "error" not in ruta else {}
        por_ruta = ruta.get("metros_por_ruta")
        distancia = por_ruta if por_ruta is not None else recta
        conocido = ruta.get("mapa_conocido_m2")
        descubierto = 0.0
        if conocido is not None:
            descubierto = 0.0 if self.ultimo_conocido is None else conocido - self.ultimo_conocido
            self.ultimo_conocido = conocido

        if datos.get("motivo") == "llego" and herramienta == "ir_a":
            self.intentos_sin_progreso = 0
        elif descubierto >= DESCUBRIMIENTO_MINIMO_M2:
            # Mapa nuevo: las rutas pueden alargarse al descubrir paredes; nueva referencia.
            self.intentos_sin_progreso = 0
            self.mejor_distancia = distancia
        elif self.mejor_distancia is None or distancia < self.mejor_distancia - MEJORA_MINIMA_M:
            self.intentos_sin_progreso = 0
        elif herramienta in HERRAMIENTAS_DE_NAVEGACION and datos.get("estado") == "fallo" and datos.get("motivo") != "sin_ruta":
            self.intentos_sin_progreso += 1
        if self.mejor_distancia is None or distancia < self.mejor_distancia:
            self.mejor_distancia = distancia

        if por_ruta is not None:
            medida = "ruta de {:.2f} m hasta {} ({:.2f} m en linea recta), mejor {:.2f} m".format(distancia, self.objetivo, recta, self.mejor_distancia)
        elif ruta.get("hay_ruta") is False:
            medida = "SIN RUTA en el mapa hasta {} ({:.2f} m en linea recta): explora con ir_a_punto".format(self.objetivo, recta)
        else:
            medida = "a {:.2f} m de {} en linea recta (mejor {:.2f} m)".format(recta, self.objetivo, self.mejor_distancia)
        mapa_txt = "" if conocido is None else " Mapa conocido {:.0f} m2 ({:+.0f}).".format(conocido, descubierto)
        return "Progreso: {}.{} Intentos de navegacion sin avance: {}/{}. Movimientos: {}/{}.".format(
            medida, mapa_txt, self.intentos_sin_progreso, self.max_sin_progreso, self.movimientos, self.max_movimientos)

    def sin_progreso(self):
        return self.intentos_sin_progreso >= self.max_sin_progreso

    def tiempo_agotado(self):
        return time.monotonic() > self.limite

    def registrar(self, tipo, **datos):
        entrada = {"t": round(time.monotonic() - self.inicio, 1), "tipo": tipo, **datos}
        self.bitacora.append(entrada)
        return entrada

    def reporte_de_respaldo(self, motivo, resumen):
        # Si el modelo no llega a reportar (limite, error), el programa reporta igual.
        pasos = ["{} {}".format(e["herramienta"], json.dumps(e.get("entrada", {}), ensure_ascii=False))
                 for e in self.bitacora if e["tipo"] == "llamada"]
        ultimo = self.ultimo_resultado_robot or {}
        return {"resultado": "fallo", "resumen": resumen, "motivo": motivo,
                "pasos_realizados": pasos,
                "recomendacion": ultimo.get("sugerencia") or "Revisar la bitacora y el estado del robot.",
                "ultimo_resultado_del_robot": {k: ultimo.get(k) for k in ("estado", "motivo", "detalle") if k in ultimo},
                "generado_por": "programa"}


def crear_herramienta_reporte(mision):
    @beta_async_tool
    async def reportar_resultado(resultado: str, resumen: str, pasos_realizados: list[str],
                                 motivo: str = "", recomendacion: str = "") -> str:
        """Reporta el final de la mision al operador. Llamala una sola vez, al terminar.

        Args:
            resultado: "exito" si la orden se cumplio y se verifico, "fallo" si no.
            resumen: Explicacion breve para el operador, en espanol.
            pasos_realizados: Lista corta de los pasos que se hicieron, en orden.
            motivo: Si fallo, el motivo tecnico (por ejemplo evasion_sin_salida) y la causa.
            recomendacion: Si fallo, que haria falta para lograrlo.
        """
        if resultado not in ("exito", "fallo"):
            return "Error: resultado debe ser 'exito' o 'fallo'."
        mision.reporte = {"resultado": resultado, "resumen": resumen, "motivo": motivo,
                          "pasos_realizados": pasos_realizados, "recomendacion": recomendacion,
                          "generado_por": "agente"}
        return "Reporte registrado. Termina tu turno sin llamar a mas herramientas."

    return reportar_resultado


def envolver_herramienta_mcp(herramienta, mision, consultar_ruta=None):
    """Aplica los limites duros antes de ejecutar una herramienta MCP y anota el resultado."""
    original = herramienta.call

    async def call(entrada):
        nombre = herramienta.name
        if mision.reporte is not None:
            return "La mision ya fue reportada; no se ejecutan mas herramientas."
        if mision.tiempo_agotado():
            return "Limite de tiempo de la mision agotado ({:.0f} s). Llama a parar y luego a reportar_resultado con fallo.".format(mision.limite_segundos) \
                if nombre != "parar" else await original(entrada)
        if nombre in HERRAMIENTAS_DE_MOVIMIENTO:
            if mision.sin_progreso():
                return ("{} intentos de navegacion seguidos sin acercarse al destino. No se ejecutan mas movimientos: "
                        "llama a parar y a reportar_resultado con fallo, explicando que bloquea el camino.").format(mision.max_sin_progreso)
            if mision.movimientos >= mision.max_movimientos:
                return "Limite de {} ordenes de movimiento alcanzado. Llama a parar y a reportar_resultado con fallo.".format(mision.max_movimientos)
            mision.movimientos += 1
        resultado = await original(entrada)
        texto = texto_de_resultado(resultado)
        try:
            datos = json.loads(texto)
            datos = datos.get("result", datos) if isinstance(datos, dict) else datos
        except (TypeError, ValueError):
            datos = None
        progreso = None
        if isinstance(datos, dict) and nombre in HERRAMIENTAS_DE_MOVIMIENTO | {"esperar_tarea"}:
            mision.ultimo_resultado_robot = datos
            mision.fijar_objetivo(nombre, datos)
            ruta = await consultar_ruta(mision.objetivo) if consultar_ruta is not None and mision.objetivo else None
            progreso = mision.actualizar_progreso(nombre, datos, ruta)
        mision.registrar("resultado", herramienta=nombre, resumen=resumir(datos if datos is not None else texto), progreso=progreso)
        imprimir("   <- {}".format(resumir(datos if datos is not None else texto)))
        if progreso is None:
            return resultado
        imprimir("      {}".format(progreso))
        if isinstance(resultado, str):
            return resultado + "\n" + progreso
        return list(resultado or []) + [{"type": "text", "text": progreso}]

    herramienta.call = call
    return herramienta


def texto_de_resultado(resultado):
    # El adaptador MCP devuelve un str o una lista de bloques de contenido.
    if isinstance(resultado, str):
        return resultado
    partes = []
    for bloque in resultado or []:
        tipo = bloque.get("type") if isinstance(bloque, dict) else getattr(bloque, "type", None)
        if tipo == "text":
            partes.append(bloque["text"] if isinstance(bloque, dict) else bloque.text)
    return "\n".join(partes)


def resumir(datos):
    if isinstance(datos, dict):
        if "error" in datos:
            return "error: {}".format(datos["error"])
        if "estado" in datos and "motivo" in datos:
            return "{} / {} | {}".format(datos.get("estado"), datos.get("motivo"), datos.get("detalle") or "")
        if "contexto" in datos:
            c = datos["contexto"]
            return "pos {} | modo {} | destino {}".format(c.get("posicion"), c.get("modo"), c.get("destino"))
        return json.dumps(datos, ensure_ascii=False)[:200]
    return str(datos)[:200]


def imprimir(texto):
    print(texto, flush=True)


async def ejecutar(orden, max_pasos, max_movimientos, minutos, esfuerzo, max_sin_progreso=3):
    mision = Mision(orden, max_movimientos, minutos * 60, max_sin_progreso)
    parametros = StdioServerParameters(command=sys.executable, args=[os.path.join(DIRECTORIO, "servidor_mcp.py")],
                                       env=dict(os.environ))
    cliente = anthropic.AsyncAnthropic()
    async with stdio_client(parametros) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            herramientas_mcp = (await sesion.list_tools()).tools
            async def consultar_ruta(destino):
                # Metros por ruta segun el mapa del robot; sin planificador o con error, None.
                try:
                    respuesta = await sesion.call_tool("distancia_por_ruta", {"destino": destino})
                    datos = json.loads(respuesta.content[0].text)
                    return datos.get("result", datos)
                except Exception:  # noqa: BLE001 - el progreso cae a linea recta
                    return None

            herramientas = [envolver_herramienta_mcp(async_mcp_tool(h, sesion), mision, consultar_ruta) for h in herramientas_mcp]
            herramientas.append(crear_herramienta_reporte(mision))

            # Sin robot no tiene sentido gastar tokens: comprobarlo antes de llamar al modelo.
            estado = await sesion.call_tool("estado_robot", {})
            texto_estado = estado.content[0].text if estado.content else "{}"
            if '"error"' in texto_estado:
                mision.reporte = mision.reporte_de_respaldo("sin_conexion_con_el_robot", json.loads(texto_estado).get("error", texto_estado))
                return mision

            # Posiciones de los destinos, para medir el progreso sin depender del modelo.
            destinos = await sesion.call_tool("listar_destinos", {})
            try:
                datos_destinos = json.loads(destinos.content[0].text)
                datos_destinos = datos_destinos.get("result", datos_destinos)
                mision.destinos = {d["nombre"]: tuple(d["posicion"]) for d in datos_destinos.get("destinos", [])}
            except (ValueError, KeyError, IndexError, AttributeError):
                mision.destinos = {}

            imprimir("ORDEN: {}".format(orden))
            runner = cliente.beta.messages.tool_runner(
                model=MODELO,
                max_tokens=16000,
                system=SISTEMA,
                tools=herramientas,
                messages=[{"role": "user", "content": orden}],
                max_iterations=max_pasos,
                thinking={"type": "adaptive"},
                output_config={"effort": esfuerzo},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            ultimo = None
            try:
                async for mensaje in runner:
                    ultimo = mensaje
                    for bloque in mensaje.content:
                        if bloque.type == "text" and bloque.text.strip():
                            mision.registrar("texto", texto=bloque.text)
                            imprimir("AGENTE: {}".format(bloque.text.strip()))
                        elif bloque.type == "tool_use":
                            mision.registrar("llamada", herramienta=bloque.name, entrada=bloque.input)
                            imprimir(" -> {}({})".format(bloque.name, json.dumps(bloque.input, ensure_ascii=False)))
                    # Sin "break": el runner termina solo cuando el modelo deja de pedir
                    # herramientas; cortarlo a mano dejaba un error al cerrar el generador.
                    if mensaje.stop_reason == "refusal":
                        mision.registrar("rechazo", detalle=str(getattr(mensaje, "stop_details", "")))
            except anthropic.APIStatusError as error:
                mision.registrar("error_api", estado=error.status_code, mensaje=str(error))
                mision.reporte = mision.reporte or mision.reporte_de_respaldo("error_api", "La API de Claude devolvio un error {}: {}".format(error.status_code, error.message))
            except anthropic.APIConnectionError as error:
                mision.registrar("error_api", mensaje=str(error))
                mision.reporte = mision.reporte or mision.reporte_de_respaldo("error_api", "No se pudo conectar con la API de Claude: {}".format(error))

            if mision.reporte is None:
                if ultimo is not None and ultimo.stop_reason == "refusal":
                    motivo, resumen = "rechazo_del_modelo", "El modelo rechazo continuar la mision."
                elif mision.tiempo_agotado():
                    motivo, resumen = "tiempo_de_mision_agotado", "Se agoto el tiempo de la mision ({} min) sin un resultado.".format(minutos)
                elif mision.sin_progreso():
                    motivo, resumen = "sin_progreso_hacia_el_destino", "{} intentos de navegacion seguidos sin acercarse al destino.".format(mision.max_sin_progreso)
                else:
                    motivo, resumen = "limite_de_pasos", "El agente uso los {} pasos permitidos sin reportar un resultado.".format(max_pasos)
                mision.reporte = mision.reporte_de_respaldo(motivo, resumen)

            # Nunca dejar el robot moviendose tras un fallo o una interrupcion.
            if mision.reporte["resultado"] != "exito":
                await sesion.call_tool("parar", {})
            if ultimo is not None:
                mision.registrar("uso", entrada=getattr(ultimo.usage, "input_tokens", None), salida=getattr(ultimo.usage, "output_tokens", None))
    return mision


def guardar(mision):
    carpeta = os.path.join(DIRECTORIO, "registros")
    os.makedirs(carpeta, exist_ok=True)
    nombre = datetime.datetime.now().strftime("mision_%Y%m%d_%H%M%S.json")
    ruta = os.path.join(carpeta, nombre)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump({"orden": mision.orden, "reporte": mision.reporte, "movimientos": mision.movimientos,
                   "bitacora": mision.bitacora}, archivo, ensure_ascii=False, indent=2, default=str)
    return ruta


def mostrar(reporte):
    imprimir("\n" + "=" * 60)
    imprimir("RESULTADO: {}".format(reporte["resultado"].upper()))
    imprimir("RESUMEN: {}".format(reporte["resumen"]))
    if reporte.get("motivo"):
        imprimir("MOTIVO: {}".format(reporte["motivo"]))
    if reporte.get("pasos_realizados"):
        imprimir("PASOS:")
        for paso in reporte["pasos_realizados"]:
            imprimir("  - {}".format(paso))
    if reporte.get("recomendacion"):
        imprimir("RECOMENDACION: {}".format(reporte["recomendacion"]))
    imprimir("=" * 60)


def main():
    cargar_env()
    parser = argparse.ArgumentParser(description="Agente autonomo del MiR100")
    parser.add_argument("orden", help='Orden en lenguaje natural, por ejemplo "Vaya al punto de entrega 3"')
    parser.add_argument("--max-pasos", type=int, default=60, help="Maximo de turnos del modelo (default 60)")
    parser.add_argument("--max-movimientos", type=int, default=30, help="Tope de ordenes de movimiento, control de costo (default 30)")
    parser.add_argument("--max-sin-progreso", type=int, default=3,
                        help="Intentos de navegacion seguidos sin acercarse al destino antes de detenerse (default 3)")
    parser.add_argument("--minutos", type=float, default=15.0, help="Tiempo maximo de la mision, reloj real (default 15)")
    parser.add_argument("--esfuerzo", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    opciones = parser.parse_args()

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        imprimir("Falta ANTHROPIC_API_KEY (en el entorno o en el .env de la raiz del repo).")
        return 1
    try:
        mision = asyncio.run(ejecutar(opciones.orden, opciones.max_pasos, opciones.max_movimientos, opciones.minutos,
                                      opciones.esfuerzo, opciones.max_sin_progreso))
    except KeyboardInterrupt:
        imprimir("\nInterrumpido. Deteniendo el robot.")
        try:
            sys.path.insert(0, DIRECTORIO)
            import herramientas
            herramientas.parar()
        except Exception:  # noqa: BLE001 - mejor esfuerzo al salir
            pass
        return 130
    mostrar(mision.reporte)
    imprimir("Bitacora: {}".format(guardar(mision)))
    return 0 if mision.reporte["resultado"] == "exito" else 2


if __name__ == "__main__":
    sys.exit(main())
