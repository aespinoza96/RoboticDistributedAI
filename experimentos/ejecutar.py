# Ejecuta el experimento: cada decisor de evasion en cada ruta, N repeticiones.
# Cada ejecucion abre Webots en modo batch y rapido; el controlador recorre la
# ruta solo, agrega una linea a resultados/resultados.jsonl y cierra Webots.
#
# Uso (desde la raiz del repo):
#   python3 experimentos/ejecutar.py                         # todo
#   python3 experimentos/ejecutar.py --decisores regla sin_jev --reps 3
#   python3 experimentos/ejecutar.py --decisores jev --rutas 1 3 --reps 5
#   python3 experimentos/ejecutar.py --decisores jev --proveedor jevai    # Jev via jevai.org
#
# Antes de lanzar ejecuciones con Jev se hace una consulta de prueba: si Jev no
# responde, esas ejecuciones se omiten (medirian la caida del servicio, no a Jev).

import argparse
import datetime
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUNDO = os.path.join(RAIZ, "mir100", "worlds", "mir100.wbt")
WEBOTS = os.environ.get("WEBOTS", "/Applications/Webots.app/Contents/MacOS/webots")
CARPETA = os.path.join(RAIZ, "experimentos", "resultados")
SALIDA = os.path.join(CARPETA, "resultados.jsonl")
LOGS = os.path.join(CARPETA, "logs")


def contar_lineas(path):
    if not os.path.exists(path):
        return 0

    with open(path, encoding="utf-8") as archivo:
        return sum(1 for _ in archivo)


def cargar_env():
    path = os.path.join(RAIZ, ".env")

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    nombre, valor = linea.split("=", 1)
                    os.environ.setdefault(nombre.strip(), valor.strip().strip('"').strip("'"))


def verificar_jev(proveedor):
    # Una consulta real con la misma pregunta que usa el robot. Devuelve (ok, detalle)
    clave = os.environ.get("TYPESAFE_API_KEY", "")
    criterios = {"LEFT": "left has clearance", "RIGHT": "right has clearance", "STOP": "unsafe"}
    estado = {"left_clearance_m": 1.8, "right_clearance_m": 0.8, "front_clearance_m": 0.7}

    try:
        if proveedor == "jevai":
            import httpx2
            respuesta = httpx2.post("https://www.jevai.org/api/v1/decisions", timeout=30,
                                    headers={"Authorization": "Bearer " + clave},
                                    json={"model": "typesafe-ai/jev", "state": estado, "questions": {
                                        "avoid_action": {"type": "choice", "instructions": "Recommend a turn.", "criteria": criterios}}})
            if respuesta.status_code != 200:
                return False, "jevai.org HTTP {}".format(respuesta.status_code)
            datos = respuesta.json()
            return datos.get("code") == 0, str(datos.get("message"))

        from typesafe_sdk import Choice, TypeSafeClient
        with TypeSafeClient(model="jev-latest") as cliente:
            cliente.system_one(state=estado, questions={"avoid_action": Choice(instructions="Recommend a turn.", criteria=criterios)})
        return True, "ok"
    except Exception as error:
        return False, "{}: {}".format(type(error).__name__, str(error)[:160])


def ejecutar(decisor, ruta, rep, max_sim_s, limite_real_s, puerto, proveedor):
    entorno = dict(os.environ)
    entorno.update({
        "JEV_PROVEEDOR": proveedor,
        "EXP_DECISOR": decisor,
        "EXP_RUTA": str(ruta),
        "EXP_REP": str(rep),
        "EXP_SALIDA": SALIDA,
        "EXP_MAX_SIM_S": str(max_sim_s),
    })

    log = os.path.join(LOGS, "{}_ruta{}_rep{}.log".format(decisor, ruta, rep))
    antes = contar_lineas(SALIDA)
    inicio = time.monotonic()

    comando = [WEBOTS, "--batch", "--mode=fast", "--minimize", "--no-rendering",
               "--stdout", "--stderr", "--port={}".format(puerto), MUNDO]

    with open(log, "w", encoding="utf-8") as archivo_log:
        proceso = subprocess.Popen(comando, env=entorno, stdout=archivo_log, stderr=subprocess.STDOUT)

        try:
            proceso.wait(timeout=limite_real_s)
            estado = "ok"
        except subprocess.TimeoutExpired:
            proceso.kill()
            proceso.wait()
            estado = "limite_real_superado"

    if contar_lineas(SALIDA) == antes:
        estado = estado if estado != "ok" else "sin_resultado"

    print("{} | {:8s} ruta {} rep {} | {:.0f} s | {}".format(
        datetime.datetime.now().strftime("%H:%M:%S"), decisor, ruta, rep, time.monotonic() - inicio, estado), flush=True)

    return estado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisores", nargs="+", default=["regla", "sin_jev", "jev"])
    parser.add_argument("--rutas", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rep-inicial", type=int, default=0)
    parser.add_argument("--max-sim-s", type=float, default=300.0)
    parser.add_argument("--limite-real-s", type=float, default=900.0)
    parser.add_argument("--puerto", type=int, default=1250)
    parser.add_argument("--proveedor", choices=["typesafe", "jevai"], default="typesafe")
    args = parser.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    cargar_env()

    if "jev" in args.decisores:
        ok, detalle = verificar_jev(args.proveedor)
        print("Verificacion de Jev ({}): {}".format(args.proveedor, "OK" if ok else "FALLA - " + detalle), flush=True)
        if not ok:
            args.decisores = [d for d in args.decisores if d != "jev"]
            print("Se omiten las ejecuciones con Jev.", flush=True)

    fallos = 0

    for rep in range(args.rep_inicial, args.rep_inicial + args.reps):
        for decisor in args.decisores:
            for ruta in args.rutas:
                if ejecutar(decisor, ruta, rep, args.max_sim_s, args.limite_real_s, args.puerto, args.proveedor) != "ok":
                    fallos += 1

    print("Terminado. Resultados en {} ({} ejecuciones sin resultado)".format(SALIDA, fallos))
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
