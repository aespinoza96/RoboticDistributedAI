"""Prueba manual del puente del robot, sin IA.

Ejemplos (con la simulacion corriendo en Webots):
    python3 agente/cli.py estado
    python3 agente/cli.py destinos
    python3 agente/cli.py ir_a destino=ENTREGA_3 --esperar
    python3 agente/cli.py girar grados=90 --esperar
    python3 agente/cli.py avanzar metros=-0.5 --esperar
    python3 agente/cli.py parar
    python3 agente/cli.py escuchar
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_cliente import RobotCliente, RobotError  # noqa: E402


def valor(texto):
    try:
        return json.loads(texto)
    except ValueError:
        return texto


def imprimir(mensaje):
    print(json.dumps(mensaje, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Comandos al MiR100 por el puente local")
    parser.add_argument("cmd", help="estado, destinos, escanear, ir_a, girar, avanzar, parar o escuchar")
    parser.add_argument("args", nargs="*", help="argumentos clave=valor")
    parser.add_argument("--esperar", action="store_true", help="esperar el evento tarea_terminada")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--puerto", type=int, default=None)
    opciones = parser.parse_args()

    argumentos = {}
    for par in opciones.args:
        if "=" not in par:
            parser.error("argumento '{}' sin '='".format(par))
        clave, texto = par.split("=", 1)
        argumentos[clave] = valor(texto)

    cliente = RobotCliente(puerto=opciones.puerto) if opciones.puerto else RobotCliente()
    try:
        with cliente as robot:
            if opciones.cmd == "escuchar":
                while True:
                    imprimir(robot.esperar_evento(timeout=None))
            resultado = robot.comando(opciones.cmd, **argumentos)
            imprimir(resultado)
            tarea_id = resultado.get("tarea_id") if isinstance(resultado, dict) else None
            if opciones.esperar and tarea_id is not None:
                while True:
                    evento = robot.esperar_evento(timeout=opciones.timeout)
                    if evento is None:
                        print("Tiempo agotado esperando el final de la tarea")
                        return 1
                    imprimir(evento)
                    if evento["evento"] == "tarea_terminada" and evento["datos"].get("tarea_id") == tarea_id:
                        return 0 if evento["datos"].get("estado") == "exito" else 2
    except RobotError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
