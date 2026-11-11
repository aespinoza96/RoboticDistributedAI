import json
import math
import os
import random
import time

# ===================================================
# MODO EXPERIMENTO: COMPARAR DECISORES DE EVASION
# ===================================================
# Se activa con variables de entorno (las pone experimentos/ejecutar.py):
#   EXP_DECISOR   "jev" | "regla" | "sin_jev"
#   EXP_RUTA      "1" | "2" | "3"  (destino ENTREGA_n, se lanza solo al arrancar)
#   EXP_SALIDA    archivo JSONL donde se agrega una linea por ejecucion
#   EXP_REP       numero de repeticion (solo se guarda)
#   EXP_MAX_SIM_S tiempo maximo de simulacion por ejecucion (s)
#
# Decisores ante un obstaculo:
#   jev     : consulta a Jev (comportamiento original del controlador)
#   regla   : lado con mas espacio, desempate hacia el destino; sin red
#   sin_jev : el controlador original sin Jev disponible (para y cancela)
# Los tres pasan por el mismo filtro de seguridad del LiDAR.

DECISORES = ("jev", "regla", "sin_jev")
CERCANIA_PELIGROSA = 0.30      # m: lectura del LiDAR por debajo cuenta como acercamiento peligroso
EMPATE_LADOS = 0.20            # m: diferencia de espacio por debajo se desempata hacia el destino
VARIACION_POSICION = 0.30      # m: desplazamiento maximo de la salida en x e y
VARIACION_ORIENTACION = 0.60   # rad: giro maximo de la orientacion de salida


def leer_configuracion():
    decisor = os.getenv("EXP_DECISOR")

    if decisor is None:
        return None

    if decisor not in DECISORES:
        raise ValueError("EXP_DECISOR debe ser uno de {}".format(DECISORES))

    return {
        "decisor": decisor,
        "ruta": int(os.getenv("EXP_RUTA", "1")),
        "rep": int(os.getenv("EXP_REP", "0")),
        "salida": os.getenv("EXP_SALIDA", "resultados.jsonl"),
        "max_sim_s": float(os.getenv("EXP_MAX_SIM_S", "300")),
    }


def variar_pose_inicial(robot_node, ruta, rep):
    # Repeticion 0 = pose del mundo. Las demas desplazan un poco la salida con una
    # semilla fija por (ruta, rep): el mismo escenario para todos los decisores
    if rep == 0:
        return None

    rng = random.Random(1000 * ruta + rep)
    dx, dy = rng.uniform(-VARIACION_POSICION, VARIACION_POSICION), rng.uniform(-VARIACION_POSICION, VARIACION_POSICION)
    dyaw = rng.uniform(-VARIACION_ORIENTACION, VARIACION_ORIENTACION)

    translation = robot_node.getField("translation")
    rotation = robot_node.getField("rotation")
    x, y, z = translation.getSFVec3f()
    axis_x, axis_y, axis_z, angle = rotation.getSFRotation()
    yaw = angle if axis_z >= 0 else -angle

    translation.setSFVec3f([x + dx, y + dy, z])
    rotation.setSFRotation([0.0, 0.0, 1.0, yaw + dyaw])
    robot_node.resetPhysics()

    return {"dx_m": round(dx, 3), "dy_m": round(dy, 3), "dyaw_rad": round(dyaw, 3)}


def decision_regla(scan, goal_angle, turn_clearance, choose_avoid_direction):
    # LEFT / RIGHT / STOP con la misma informacion que recibe Jev
    candidates = [side for side in ("LEFT", "RIGHT") if turn_clearance(scan, side)]

    if not candidates:
        return "STOP"

    if len(candidates) == 2 and abs(scan["left"] - scan["right"]) < EMPATE_LADOS:
        return "LEFT" if goal_angle >= 0 else "RIGHT"

    preferred = choose_avoid_direction(scan["left"], scan["right"])

    return preferred if preferred in candidates else candidates[0]


class RegistroExperimento:
    def __init__(self, config):
        self.config = config
        self.iniciado = False
        self.terminado = False

        self.t_inicio_sim = 0.0
        self.t_inicio_real = 0.0
        self.posicion_anterior = None
        self.distancia_recorrida = 0.0
        self.distancia_minima = float("inf")
        self.pasos_peligrosos = 0
        self.t_esperando_jev_sim = 0.0
        self.t_parado_persona_sim = 0.0
        self.paradas_seguridad = 0
        self.decisiones = []
        self.ultimo_motivo = None
        self.variacion_inicial = None

    def iniciar(self, t_sim, posicion, distancia_objetivo):
        self.iniciado = True
        self.t_inicio_sim = t_sim
        self.t_inicio_real = time.monotonic()
        self.posicion_anterior = posicion
        self.distancia_inicial = distancia_objetivo

    def paso(self, posicion, scan, modo, dt, parada_seguridad):
        if not self.iniciado or self.terminado:
            return

        self.distancia_recorrida += math.hypot(posicion[0] - self.posicion_anterior[0],
                                               posicion[1] - self.posicion_anterior[1])
        self.posicion_anterior = posicion

        lecturas = [scan[k] for k in ("left", "front_left", "front", "front_right", "right", "back") if scan[k] > 0.0]

        if lecturas:
            minima = min(lecturas)
            self.distancia_minima = min(self.distancia_minima, minima)

            if minima < CERCANIA_PELIGROSA:
                self.pasos_peligrosos += 1

        if modo == "JEV_WAIT":
            self.t_esperando_jev_sim += dt
        elif modo == "WAIT_PERSON":
            self.t_parado_persona_sim += dt

        if parada_seguridad:
            self.paradas_seguridad += 1

    def decision(self, decisor, propuesta, confianza, latencia_s, aceptada, motivo, regla):
        self.decisiones.append({
            "decisor": decisor,
            "propuesta": propuesta,
            "confianza": confianza,
            "latencia_s": round(latencia_s, 3),
            "aceptada": aceptada,
            "motivo_rechazo": motivo,
            "regla_habria_elegido": regla,
        })

    def cancelar_por(self, motivo):
        self.ultimo_motivo = motivo

    def terminar(self, resultado, t_sim, distancia_objetivo):
        if self.terminado:
            return

        self.terminado = True
        t_sim_total = t_sim - self.t_inicio_sim
        latencia_total = sum(d["latencia_s"] for d in self.decisiones)

        registro = {
            "decisor": self.config["decisor"],
            "ruta": self.config["ruta"],
            "rep": self.config["rep"],
            "variacion_inicial": self.variacion_inicial,
            "resultado": resultado,
            "motivo": self.ultimo_motivo if resultado == "cancelada" else None,
            "t_sim_s": round(t_sim_total, 2),
            # La simulacion corre en modo rapido: la espera a Jev cuenta en tiempo de
            # simulacion mas de lo que dura en realidad. Tiempo efectivo = tiempo sin
            # esas esperas + la latencia real de cada consulta
            "t_efectivo_s": round(t_sim_total - self.t_esperando_jev_sim + latencia_total, 2),
            "t_real_s": round(time.monotonic() - self.t_inicio_real, 2),
            "distancia_inicial_m": round(self.distancia_inicial, 2),
            "distancia_final_m": round(distancia_objetivo, 2),
            "recorrido_m": round(self.distancia_recorrida, 2),
            "distancia_minima_lidar_m": round(self.distancia_minima, 3) if math.isfinite(self.distancia_minima) else None,
            "pasos_peligrosos": self.pasos_peligrosos,
            "paradas_seguridad": self.paradas_seguridad,
            "t_parado_persona_s": round(self.t_parado_persona_sim, 2),
            "t_esperando_jev_sim_s": round(self.t_esperando_jev_sim, 2),
            "latencia_jev_total_s": round(latencia_total, 3),
            "n_decisiones": len(self.decisiones),
            "decisiones": self.decisiones,
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.config["salida"])), exist_ok=True)

        with open(self.config["salida"], "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

        print("EXPERIMENTO TERMINADO: {} | {} | ruta {} | {:.1f} s".format(
            resultado, self.config["decisor"], self.config["ruta"], t_sim_total))
