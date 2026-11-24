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
HERRAMIENTAS_DE_MOVIMIENTO = {"ir_a", "girar", "avanzar"}

SISTEMA = """\
Eres el agente de mision de un robot movil MiR100 en una fabrica simulada (Webots).
Recibes una orden de un operador y debes cumplirla de forma autonoma usando las
herramientas del robot, o explicar con claridad por que no se puede.

Como trabajar:
1. Entiende la orden y el estado: usa estado_robot y, si hace falta, listar_destinos.
2. Actua: las herramientas de movimiento esperan a que la tarea termine y devuelven
   estado (exito / fallo / cancelada), motivo, detalle, eventos y una sugerencia.
3. Si una tarea falla, analiza el motivo y el contexto, usa escanear_entorno y prueba
   una recuperacion corta (por ejemplo retroceder con avanzar negativo y girar hacia el
   lado con mas espacio) antes de reintentar. Cambia de estrategia entre intentos: no
   repitas exactamente lo que ya fallo.
4. Verifica el resultado con estado_robot antes de darlo por cumplido.

Limites:
- La capa de seguridad del robot siempre tiene la ultima palabra. Si una maniobra queda
  bloqueada por seguridad, no insistas en la misma direccion.
- Como maximo 3 ciclos de recuperacion para el mismo problema. Si no hay progreso,
  deten el robot con parar y reporta el fallo.
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

    def __init__(self, orden, max_movimientos, limite_segundos):
        self.orden = orden
        self.max_movimientos = max_movimientos
        self.limite = time.monotonic() + limite_segundos
        self.limite_segundos = limite_segundos
        self.inicio = time.monotonic()
        self.movimientos = 0
        self.reporte = None
        self.bitacora = []
        self.ultimo_resultado_robot = None

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


def envolver_herramienta_mcp(herramienta, mision):
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
        if isinstance(datos, dict) and nombre in HERRAMIENTAS_DE_MOVIMIENTO | {"esperar_tarea"}:
            mision.ultimo_resultado_robot = datos
        mision.registrar("resultado", herramienta=nombre, resumen=resumir(datos if datos is not None else texto))
        imprimir("   <- {}".format(resumir(datos if datos is not None else texto)))
        return resultado

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


async def ejecutar(orden, max_pasos, max_movimientos, minutos, esfuerzo):
    mision = Mision(orden, max_movimientos, minutos * 60)
    parametros = StdioServerParameters(command=sys.executable, args=[os.path.join(DIRECTORIO, "servidor_mcp.py")],
                                       env=dict(os.environ))
    cliente = anthropic.AsyncAnthropic()
    async with stdio_client(parametros) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            herramientas_mcp = (await sesion.list_tools()).tools
            herramientas = [envolver_herramienta_mcp(async_mcp_tool(h, sesion), mision) for h in herramientas_mcp]
            herramientas.append(crear_herramienta_reporte(mision))

            # Sin robot no tiene sentido gastar tokens: comprobarlo antes de llamar al modelo.
            estado = await sesion.call_tool("estado_robot", {})
            texto_estado = estado.content[0].text if estado.content else "{}"
            if '"error"' in texto_estado:
                mision.reporte = mision.reporte_de_respaldo("sin_conexion_con_el_robot", json.loads(texto_estado).get("error", texto_estado))
                return mision

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
                    if mensaje.stop_reason == "refusal":
                        mision.registrar("rechazo", detalle=str(getattr(mensaje, "stop_details", "")))
                        break
                    if mision.reporte is not None and mensaje.stop_reason != "tool_use":
                        break
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
    parser.add_argument("--max-pasos", type=int, default=30, help="Maximo de turnos del modelo (default 30)")
    parser.add_argument("--max-movimientos", type=int, default=12, help="Maximo de ordenes de movimiento (default 12)")
    parser.add_argument("--minutos", type=float, default=15.0, help="Tiempo maximo de la mision, reloj real (default 15)")
    parser.add_argument("--esfuerzo", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    opciones = parser.parse_args()

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        imprimir("Falta ANTHROPIC_API_KEY (en el entorno o en el .env de la raiz del repo).")
        return 1
    try:
        mision = asyncio.run(ejecutar(opciones.orden, opciones.max_pasos, opciones.max_movimientos, opciones.minutos, opciones.esfuerzo))
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
