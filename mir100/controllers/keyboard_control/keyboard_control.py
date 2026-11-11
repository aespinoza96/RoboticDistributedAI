from controller import Supervisor, Keyboard
from ultralytics import YOLO
import numpy as np
import cv2
import math
import os
import threading
import queue
import time


def load_env_file():
    # Busca un .env subiendo desde este archivo (raiz del repo) y carga sus
    # variables sin sobrescribir las que ya existan en el entorno.
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        path = os.path.join(directory, ".env")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as env_file:
                for line in env_file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    name = name.strip()
                    if name.startswith("export "):
                        name = name[len("export "):].strip()
                    value = value.strip().strip('"').strip("'")
                    if name and value and name not in os.environ:
                        os.environ[name] = value
            return path
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


load_env_file()

# Modo experimento (variables EXP_*, ver experimentos/ejecutar.py); None = uso normal
from experimento import RegistroExperimento, decision_regla, leer_configuracion, variar_pose_inicial
EXPERIMENTO = leer_configuracion()

try:
    from typesafe_sdk import TypeSafeClient, Choice
    JEV_SDK_AVAILABLE = True
except ImportError:
    JEV_SDK_AVAILABLE = False

# ===================================================
# CONFIGURACION - AJUSTAR EN TU ESCENARIO
# ===================================================
WHEEL_RADIUS = 0.0625
DISTANCE_TO_CENTER = 0.2226
SPEED_MAX = 0.95
SPEED_MIN = -0.30
ANGULAR_SPEED_MAX = 0.30
SPEED_INCREMENT = 0.05
ANGULAR_SPEED_INCREMENT = 0.05

WORLD_UP_AXIS = "Z"                 # Tu mundo parece usar Z como vertical.
ROBOT_FORWARD_AXIS = "+X"          # Verificar que +X sea el frente real del MiR100.
TURN_SIGN = 1.0                    # Cambiar a -1.0 SOLO si el robot gira al reves.
LIDAR_SWAP_SIDES = False           # Cambiar a True SOLO si izq/der del LiDAR estan invertidas.

DESTINATIONS = {ord("1"): "ENTREGA_1", ord("2"): "ENTREGA_2", ord("3"): "ENTREGA_3"}
ARRIVAL_DISTANCE = 0.30
NAVIGATION_SPEED = 0.23
NAVIGATION_TURN_SPEED = 0.25
NAVIGATION_ANGLE_TOLERANCE = 0.12

# Distancias en metros. Deben validarse segun el ancho real del robot y su entorno.
SAFE_DISTANCE_FRONT = 1.00
SAFE_DISTANCE_BACK = 0.80
FRONT_CLEAR_DISTANCE = 1.35
SIDE_MIN_DISTANCE = 0.65
SIDE_DESIRED_DISTANCE = 0.90
SIDE_KP = 0.32
AVOID_FORWARD_SPEED = 0.13
AVOID_TURN_SPEED = 0.25
AVOID_MIN_TRAVEL = 1.00            # Distancia minima recorrida antes de volver a apuntar al destino.
AVOID_MAX_SECONDS = 45.0          # No insistir indefinidamente ante una ruta bloqueada.
AVOID_HEADING_TOLERANCE = 0.30
MAX_AVOID_ATTEMPTS = 8

YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.55
YOLO_EVERY_N_STEPS = 3
PERSON_CLEAR_DETECTIONS = 5       # 5 inferencias seguidas sin persona antes de reanudar.
LOG_EVERY_STEPS = 30

# JEV (TypeSafe): asesor de decisiones, nunca control directo de motores.
JEV_ENABLED = True
JEV_MIN_CONFIDENCE = 0.60
JEV_REQUEST_TIMEOUT = 8.0      # Segundos de reloj real, sin bloquear Webots.
JEV_SIDE_CLEARANCE = 0.80      # Verificar con el ancho y volumen del robot.
SAFE_DISTANCE_SIDE = 0.60      # Margen inicial conservador; calibrar para el robot.
SAFE_DISTANCE_DIAGONAL = 0.95  # No atravesar esquinas de maquinaria.
TURN_CLEARANCE_MIN = 0.65      # Distancia lateral minima para permitir una rotacion.
LIDAR_UNKNOWN_IS_BLOCKED = True
LOG_SECTOR_INTERVAL = 30

MODE_GOAL = "GOAL"
MODE_AVOID_TURN = "AVOID_TURN"
MODE_AVOID_FOLLOW = "AVOID_FOLLOW"
MODE_WAIT_PERSON = "WAIT_PERSON"
MODE_JEV_WAIT = "JEV_WAIT"

# ===================================================
# FUNCIONES
# ===================================================
def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def axis_vector(axis):
    axes = {"+X": [1.0, 0.0, 0.0], "-X": [-1.0, 0.0, 0.0], "+Y": [0.0, 1.0, 0.0], "-Y": [0.0, -1.0, 0.0], "+Z": [0.0, 0.0, 1.0], "-Z": [0.0, 0.0, -1.0]}
    return np.array(axes[axis], dtype=float)


def planar_position(node):
    position = np.array(node.getPosition(), dtype=float)
    up = axis_vector("+" + WORLD_UP_AXIS)
    return position - np.dot(position, up) * up


def goal_geometry(robot_node, destination_node):
    displacement = planar_position(destination_node) - planar_position(robot_node)
    distance = float(np.linalg.norm(displacement))
    if distance < 0.0001:
        return distance, 0.0
    up = axis_vector("+" + WORLD_UP_AXIS)
    orientation = np.array(robot_node.getOrientation(), dtype=float).reshape((3, 3))
    forward = orientation.dot(axis_vector(ROBOT_FORWARD_AXIS))
    forward -= np.dot(forward, up) * up
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 0.0001:
        raise ValueError("ROBOT_FORWARD_AXIS apunta en vertical: revisa el eje delantero")
    forward /= forward_norm
    desired = displacement / distance
    angle_error = math.atan2(float(np.dot(up, np.cross(forward, desired))), float(np.dot(forward, desired)))
    return distance, angle_error


def normalize_ranges(raw, maximum):
    values = np.array(raw, dtype=float)
    values[np.isposinf(values)] = maximum
    values[~np.isfinite(values)] = 0.0
    values[(values <= 0.0) | (values > maximum)] = 0.0
    return values


def sector_distance(values, start_fraction, end_fraction, method="percentile"):
    total = len(values)
    part = values[int(total * start_fraction):max(int(total * end_fraction), int(total * start_fraction) + 1)]
    valid = part[part > 0.0]
    if len(valid) < max(1, int(len(part) * 0.25)):
        return 0.0
    if method == "minimum":
        return float(np.min(valid))
    return float(np.percentile(valid, 20))


def read_lidar_distances(front_lidar, back_lidar):
    # El orden de los sectores debe validarse visualmente en Webots.
    # Nunca se asume que una zona sin rayos validos este despejada.
    try:
        front_values = normalize_ranges(front_lidar.getLayerRangeImage(0), front_lidar.getMaxRange())
        back_values = normalize_ranges(back_lidar.getLayerRangeImage(0), back_lidar.getMaxRange())
    except Exception as error:
        print("ERROR LIDAR: {}".format(error))
        return {"left": 0.0, "front_left": 0.0, "front": 0.0, "front_right": 0.0, "right": 0.0, "back": 0.0, "ok": False}

    if len(front_values) == 0 or len(back_values) == 0:
        return {"left": 0.0, "front_left": 0.0, "front": 0.0, "front_right": 0.0, "right": 0.0, "back": 0.0, "ok": False}

    data = {
        "left": sector_distance(front_values, 0.06, 0.23),
        "front_left": sector_distance(front_values, 0.23, 0.41),
        "front": sector_distance(front_values, 0.41, 0.59, "minimum"),
        "front_right": sector_distance(front_values, 0.59, 0.77),
        "right": sector_distance(front_values, 0.77, 0.94),
        "back": sector_distance(back_values, 0.35, 0.65, "minimum")
    }

    if LIDAR_SWAP_SIDES:
        data["left"], data["right"] = data["right"], data["left"]
        data["front_left"], data["front_right"] = data["front_right"], data["front_left"]

    data["ok"] = all(data[k] > 0.0 for k in ("left", "front_left", "front", "front_right", "right", "back"))
    return data


def forward_corridor_clear(scan):
    return scan["ok"] and scan["front"] >= SAFE_DISTANCE_FRONT and scan["front_left"] >= SAFE_DISTANCE_DIAGONAL and scan["front_right"] >= SAFE_DISTANCE_DIAGONAL and scan["left"] >= SAFE_DISTANCE_SIDE and scan["right"] >= SAFE_DISTANCE_SIDE


def turn_clearance(scan, direction):
    # Una medida del costado no garantiza que el robot pueda girar: exigir tambien diagonal.
    if not scan["ok"]:
        return False
    if direction == "LEFT":
        return scan["left"] >= TURN_CLEARANCE_MIN and scan["front_left"] >= TURN_CLEARANCE_MIN
    if direction == "RIGHT":
        return scan["right"] >= TURN_CLEARANCE_MIN and scan["front_right"] >= TURN_CLEARANCE_MIN
    return False


def scan_text(scan):
    return "I {:.2f} DI {:.2f} F {:.2f} DD {:.2f} D {:.2f}".format(scan["left"], scan["front_left"], scan["front"], scan["front_right"], scan["right"])


def choose_avoid_direction(left, right):
    if left < SIDE_MIN_DISTANCE and right < SIDE_MIN_DISTANCE:
        return None
    if left >= right:
        return "LEFT"
    return "RIGHT"


def follow_side_omega(direction, left, right, goal_angle):
    # Girar a derecha deja normalmente el obstaculo a la IZQUIERDA.
    if direction == "RIGHT":
        side = left
        omega = SIDE_KP * (side - SIDE_DESIRED_DISTANCE)
    else:
        side = right
        omega = SIDE_KP * (SIDE_DESIRED_DISTANCE - side)
    # Pequeno sesgo hacia el objetivo; el LiDAR sigue siendo prioritario.
    omega += 0.05 * clamp(goal_angle, -1.0, 1.0)
    return clamp(omega, -NAVIGATION_TURN_SPEED, NAVIGATION_TURN_SPEED)


# ===================================================
# CONSULTAS JEV EN SEGUNDO HILO (SIN ACCESO A WEBOTS EN EL HILO)
# ===================================================
JEV_INSTRUCTIONS = "Recommend one short exploratory turn, not a complete route. Prefer the side with adequate side AND forward-diagonal clearance. Use STOP if no safe candidate exists or observations are uncertain. The robot safety controller has final authority; your recommendation never bypasses safety constraints."
JEV_CRITERIA = {
    "LEFT": "Turn in place left to search for a free path, only if the measured left sector provides adequate clearance",
    "RIGHT": "Turn in place right to search for a free path, only if the measured right sector provides adequate clearance",
    "STOP": "Stop because both sides are narrow, a person is present, or measurements are too uncertain"
}

# Proveedor de Jev: "typesafe" (SDK oficial, api.typesafe.ai) o "jevai"
# (https://www.jevai.org/api/v1/decisions, claves personales de jevai.org).
# La misma pregunta y la misma clave (TYPESAFE_API_KEY) en ambos casos.
JEV_PROVIDER = os.getenv("JEV_PROVEEDOR", "typesafe")
JEVAI_URL = "https://www.jevai.org/api/v1/decisions"


def ask_jevai(observation):
    import httpx2
    body = {"model": "typesafe-ai/jev", "state": observation, "questions": {
        "avoid_action": {"type": "choice", "instructions": JEV_INSTRUCTIONS, "criteria": JEV_CRITERIA}}}
    response = httpx2.post(JEVAI_URL, json=body, timeout=JEV_REQUEST_TIMEOUT,
                           headers={"Authorization": "Bearer " + os.getenv("TYPESAFE_API_KEY", "")})
    if response.status_code != 200:
        raise RuntimeError("jevai.org HTTP {}".format(response.status_code))
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError("jevai.org: {}".format(payload.get("message")))
    answer = payload["data"]["answers"]["avoid_action"]
    return str(answer["choice"]).upper(), float(answer["confidence"])


def ask_jev_worker(request_id, observation, response_queue):
    try:
        if JEV_PROVIDER == "jevai":
            choice, confidence = ask_jevai(observation)
        else:
            with TypeSafeClient(model="jev-latest") as client:
                response = client.system_one(state=observation, questions={
                    "avoid_action": Choice(instructions=JEV_INSTRUCTIONS, criteria=JEV_CRITERIA)
                })
            answer = response.choices["avoid_action"]
            choice, confidence = str(answer.choice).upper(), float(answer.confidence)
        response_queue.put((request_id, choice, confidence, None))
    except Exception as error:
        response_queue.put((request_id, None, 0.0, "{}: {}".format(type(error).__name__, error)))


def submit_jev_request(request_id, distance, angle, scan, destination, response_queue):
    observation = {
        "robot": "MiR100 differential drive in Webots",
        "destination": destination,
        "goal_distance_m": round(distance, 2),
        "goal_angle_rad": round(angle, 3),
        "left_clearance_m": round(scan["left"], 2),
        "front_left_clearance_m": round(scan["front_left"], 2),
        "front_clearance_m": round(scan["front"], 2),
        "front_right_clearance_m": round(scan["front_right"], 2),
        "right_clearance_m": round(scan["right"], 2),
        "safe_left_turn": turn_clearance(scan, "LEFT"),
        "safe_right_turn": turn_clearance(scan, "RIGHT"),
        "minimum_side_clearance_m": JEV_SIDE_CLEARANCE,
        "front_safety_distance_m": SAFE_DISTANCE_FRONT,
        "observation": "Static industrial machinery obstructs forward progress. Need a short local maneuver. A side reading does not prove a traversable corridor.",
        "person_detected": False
    }
    thread = threading.Thread(target=ask_jev_worker, args=(request_id, observation, response_queue), daemon=True)
    thread.start()


# ===================================================
# INICIALIZACION
# ===================================================
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()
if robot_node is None:
    raise RuntimeError("Activa supervisor TRUE en el nodo principal del MiR100")

motor_left_wheel = robot.getDevice("middle_left_wheel_joint")
motor_right_wheel = robot.getDevice("middle_right_wheel_joint")
motor_left_wheel.setPosition(float("inf"))
motor_right_wheel.setPosition(float("inf"))
motor_left_wheel.setVelocity(0.0)
motor_right_wheel.setVelocity(0.0)

front_lidar = robot.getDevice("front_lidar")
back_lidar = robot.getDevice("back_lidar")
front_lidar.enable(time_step)
back_lidar.enable(time_step)

camera = robot.getDevice("camera")
camera.enable(time_step)
camera_width = camera.getWidth()
camera_height = camera.getHeight()

keyboard = Keyboard()
keyboard.enable(time_step)
model = YOLO(YOLO_MODEL)

destination_nodes = {}
for key, destination_def in DESTINATIONS.items():
    node = robot.getFromDef(destination_def)
    if node is not None:
        destination_nodes[key] = node
        print("DESTINO DISPONIBLE: {} -> {}".format(destination_def, node.getPosition()))
    else:
        print("DESTINO NO ENCONTRADO: {}".format(destination_def))

navigation_active = False
current_destination = None
current_destination_name = None
mode = MODE_GOAL
saved_mode = MODE_GOAL
person_detected = False
person_confidence = 0.0
person_clear_count = 0
annotated_frame = None
step_counter = 0
avoid_direction = None
avoid_attempts = 0
avoid_started = 0.0
avoid_start_position = None
manual_speed = 0.0
manual_omega = 0.0
jev_results = queue.Queue()
jev_request_id = 0
jev_started_at = 0.0
jev_pending = False
jev_blocked_reason = None

if JEV_ENABLED and not JEV_SDK_AVAILABLE:
    print("ADVERTENCIA: instala typesafe-sdk en el Python de Webots. Sin Jev, los destinos se detendran ante obstaculos.")
if JEV_ENABLED and not os.getenv("TYPESAFE_API_KEY"):
    print("ADVERTENCIA: falta TYPESAFE_API_KEY. Se detendra ante obstaculos.")

print("\nMiR100: YOLO + LiDAR + NAVEGACION LOCAL REACTIVA")
print("Teclas 1/2/3: destino | 0 o ESPACIO: cancelar | Flechas: control manual")
print("JEV: asesor para elegir giro cuando hay obstaculos (no planificador global)")
print("PRECAUCION: prueba inicial en simulacion despejada, velocidad baja\n")

registro = RegistroExperimento(EXPERIMENTO) if EXPERIMENTO else None
salida_enviada = False
jev_regla_esperada = None
if registro is not None:
    print("MODO EXPERIMENTO: decisor {} | ruta {} | rep {}".format(EXPERIMENTO["decisor"], EXPERIMENTO["ruta"], EXPERIMENTO["rep"]))
    registro.variacion_inicial = variar_pose_inicial(robot_node, EXPERIMENTO["ruta"], EXPERIMENTO["rep"])

# ===================================================
# BUCLE PRINCIPAL
# ===================================================
while robot.step(time_step) != -1:
    step_counter += 1
    scan = read_lidar_distances(front_lidar, back_lidar)
    left_distance = scan["left"]
    front_left_distance = scan["front_left"]
    front_distance = scan["front"]
    front_right_distance = scan["front_right"]
    right_distance = scan["right"]
    back_distance = scan["back"]
    lidar_ok = scan["ok"]

    # YOLO analiza todas las clases COCO: las personas activan STOP; los demas
    # objetos solo se etiquetan. El LiDAR determina si el espacio es transitable.
    image = camera.getImage()
    if image:
        frame = np.frombuffer(image, dtype=np.uint8).reshape((camera_height, camera_width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if step_counter % YOLO_EVERY_N_STEPS == 0:
            results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
            boxes = results[0].boxes if results else None
            person_detected = False
            person_confidence = 0.0
            if boxes is not None:
                for detected_box in boxes:
                    class_id = int(detected_box.cls[0])
                    confidence = float(detected_box.conf[0])
                    if class_id == 0:
                        person_detected = True
                        person_confidence = max(person_confidence, confidence)
            annotated_frame = results[0].plot() if results else frame
            if person_detected:
                person_clear_count = 0
            else:
                person_clear_count += 1
        if registro is None:
            cv2.imshow("MiR100 - YOLO todas las clases", annotated_frame if annotated_frame is not None else frame)
            cv2.waitKey(1)

    # Leer todas las teclas; ultimo evento tiene prioridad.
    key = keyboard.getKey()
    while True:
        next_key = keyboard.getKey()
        if next_key == -1:
            break
        key = next_key

    # Experimento: lanzar la ruta sola tras unos pasos (sensores y YOLO ya listos)
    if registro is not None and not registro.iniciado and step_counter == 10:
        key = ord(str(EXPERIMENTO["ruta"]))

    if key in destination_nodes:
        jev_request_id += 1
        jev_pending = False
        current_destination = destination_nodes[key]
        current_destination_name = DESTINATIONS[key]
        navigation_active = True
        mode = MODE_WAIT_PERSON if person_detected else MODE_GOAL
        saved_mode = MODE_GOAL
        avoid_attempts = 0
        manual_speed = 0.0
        manual_omega = 0.0
        print("\nINICIANDO RUTA: {}".format(current_destination_name))
        if registro is not None and not registro.iniciado:
            registro.iniciar(robot.getTime(), planar_position(robot_node), goal_geometry(robot_node, current_destination)[0])
    elif key == ord("0") or key == ord(" "):
        jev_request_id += 1
        jev_pending = False
        navigation_active = False
        current_destination = None
        current_destination_name = None
        mode = MODE_GOAL
        manual_speed = 0.0
        manual_omega = 0.0
        print("NAVEGACION CANCELADA: STOP")
    elif not navigation_active:
        if key == Keyboard.UP:
            manual_speed = min(SPEED_MAX, manual_speed + SPEED_INCREMENT)
        elif key == Keyboard.DOWN:
            manual_speed = max(SPEED_MIN, manual_speed - SPEED_INCREMENT)
        elif key == Keyboard.LEFT:
            manual_omega = min(ANGULAR_SPEED_MAX, manual_omega + ANGULAR_SPEED_INCREMENT)
        elif key == Keyboard.RIGHT:
            manual_omega = max(-ANGULAR_SPEED_MAX, manual_omega - ANGULAR_SPEED_INCREMENT)

    target_speed = manual_speed if not navigation_active else 0.0
    target_omega = manual_omega if not navigation_active else 0.0

    # Persona: se conservan los datos de evasion, se frena y se reanuda despues.
    if person_detected and mode != MODE_WAIT_PERSON:
        saved_mode = mode
        mode = MODE_WAIT_PERSON
        manual_speed = 0.0
        manual_omega = 0.0
        print("PERSONA DETECTADA (confianza {:.2f}): STOP".format(person_confidence))

    if mode == MODE_WAIT_PERSON:
        target_speed = 0.0
        target_omega = 0.0
        if not person_detected and person_clear_count >= PERSON_CLEAR_DETECTIONS:
            mode = saved_mode
            if mode in (MODE_AVOID_TURN, MODE_AVOID_FOLLOW):
                avoid_started = robot.getTime()
            print("PERSONA DESPEJADA: REANUDANDO")

    elif navigation_active and current_destination is not None:
        distance, goal_angle = goal_geometry(robot_node, current_destination)
        if distance <= ARRIVAL_DISTANCE:
            print("\nDESTINO ALCANZADO: {} | Distancia {:.2f} m".format(current_destination_name, distance))
            if registro is not None:
                registro.terminar("llegada", robot.getTime(), distance)
            navigation_active = False
            current_destination = None
            current_destination_name = None
            mode = MODE_GOAL
            manual_speed = 0.0
            manual_omega = 0.0
            target_speed = 0.0
            target_omega = 0.0
        elif not lidar_ok:
            target_speed = 0.0
            target_omega = 0.0
            if step_counter % LOG_EVERY_STEPS == 0:
                print("SEGURIDAD: LiDAR sin lecturas suficientes, STOP")
        elif mode == MODE_JEV_WAIT:
            # En espera de red: motores inmoviles y sin bloquear robot.step().
            target_speed = 0.0
            target_omega = 0.0
            decision = None
            while not jev_results.empty():
                result = jev_results.get_nowait()
                if result[0] == jev_request_id:
                    decision = result
            if decision is not None:
                jev_pending = False
                _, proposed, confidence, error = decision
                if error is not None:
                    print("JEV ERROR: {} | STOP".format(error))
                else:
                    print("JEV RESPUESTA: {} | confianza {:.2f}".format(proposed, confidence))
                side_space = left_distance if proposed == "LEFT" else right_distance if proposed == "RIGHT" else 0.0
                if error is not None:
                    jev_motivo = "error"
                elif confidence < JEV_MIN_CONFIDENCE:
                    jev_motivo = "baja_confianza"
                elif proposed not in ("LEFT", "RIGHT"):
                    jev_motivo = "jev_stop"
                elif side_space < JEV_SIDE_CLEARANCE or not turn_clearance(scan, proposed):
                    jev_motivo = "filtro_lidar"
                else:
                    jev_motivo = None
                if registro is not None:
                    registro.decision("jev", proposed, confidence, time.monotonic() - jev_started_at,
                                      jev_motivo is None, jev_motivo, jev_regla_esperada)
                if jev_motivo is not None:
                    print("JEV: decision rechazada por seguridad o baja confianza. STOP")
                    if registro is not None:
                        registro.cancelar_por("jev_" + jev_motivo if not jev_motivo.startswith("jev_") else jev_motivo)
                    navigation_active = False
                    current_destination = None
                    current_destination_name = None
                    mode = MODE_GOAL
                else:
                    avoid_direction = proposed
                    avoid_attempts += 1
                    mode = MODE_AVOID_TURN
                    avoid_started = robot.getTime()
                    avoid_start_position = planar_position(robot_node)
                    print("JEV APROBADO POR LIDAR: GIRAR {}".format(avoid_direction))
            elif time.monotonic() - jev_started_at >= JEV_REQUEST_TIMEOUT:
                jev_pending = False
                jev_request_id += 1  # Descartar respuesta tardia.
                print("JEV TIMEOUT: STOP (sin decision vigente)")
                if registro is not None:
                    registro.decision("jev", None, None, JEV_REQUEST_TIMEOUT, False, "timeout", jev_regla_esperada)
                    registro.cancelar_por("jev_timeout")
                navigation_active = False
                current_destination = None
                current_destination_name = None
                mode = MODE_GOAL
        elif mode == MODE_GOAL:
            # Girar hacia destino sin avanzar hasta que este orientado.
            if abs(goal_angle) > NAVIGATION_ANGLE_TOLERANCE:
                target_speed = 0.0
                target_omega = clamp(goal_angle * 1.2, -NAVIGATION_TURN_SPEED, NAVIGATION_TURN_SPEED)
            elif forward_corridor_clear(scan):
                target_speed = min(NAVIGATION_SPEED, max(0.06, 0.5 * distance))
                target_omega = clamp(goal_angle * 0.5, -NAVIGATION_TURN_SPEED, NAVIGATION_TURN_SPEED)
            else:
                # Obstruccion: Jev puede recomendar un giro, pero no ejecuta motores.
                target_speed = 0.0
                target_omega = 0.0
                if avoid_attempts >= MAX_AVOID_ATTEMPTS or not (turn_clearance(scan, "LEFT") or turn_clearance(scan, "RIGHT")):
                    print("NO HAY ESPACIO LATERAL SUFICIENTE / LIMITE DE EVASIONES: STOP")
                    if registro is not None:
                        registro.cancelar_por("sin_espacio_o_limite")
                    navigation_active = False
                    current_destination = None
                    current_destination_name = None
                elif registro is not None and EXPERIMENTO["decisor"] == "regla":
                    # Misma informacion y mismo filtro que Jev, decidido localmente
                    proposed = decision_regla(scan, goal_angle, turn_clearance, choose_avoid_direction)
                    side_space = left_distance if proposed == "LEFT" else right_distance if proposed == "RIGHT" else 0.0
                    if proposed not in ("LEFT", "RIGHT") or side_space < JEV_SIDE_CLEARANCE or not turn_clearance(scan, proposed):
                        print("REGLA: sin lado seguro. STOP")
                        registro.decision("regla", proposed, None, 0.0, False, "filtro_lidar", proposed)
                        registro.cancelar_por("regla_filtro_lidar")
                        navigation_active = False
                        current_destination = None
                        current_destination_name = None
                    else:
                        registro.decision("regla", proposed, None, 0.0, True, None, proposed)
                        avoid_direction = proposed
                        avoid_attempts += 1
                        mode = MODE_AVOID_TURN
                        avoid_started = robot.getTime()
                        avoid_start_position = planar_position(robot_node)
                        print("REGLA APROBADA POR LIDAR: GIRAR {}".format(avoid_direction))
                elif (registro is not None and EXPERIMENTO["decisor"] == "sin_jev") or not JEV_ENABLED or not JEV_SDK_AVAILABLE or not os.getenv("TYPESAFE_API_KEY"):
                    print("JEV NO DISPONIBLE: STOP (se requiere Jev para la evasion en este prototipo)")
                    if registro is not None:
                        registro.cancelar_por("jev_no_disponible")
                    navigation_active = False
                    current_destination = None
                    current_destination_name = None
                else:
                    jev_request_id += 1
                    jev_pending = True
                    jev_started_at = time.monotonic()
                    mode = MODE_JEV_WAIT
                    jev_regla_esperada = decision_regla(scan, goal_angle, turn_clearance, choose_avoid_direction)
                    submit_jev_request(jev_request_id, distance, goal_angle, scan, current_destination_name, jev_results)
                    print("JEV CONSULTANDO | {} | destino {}".format(scan_text(scan), current_destination_name))
        elif mode == MODE_AVOID_TURN:
            target_speed = 0.0
            target_omega = AVOID_TURN_SPEED if avoid_direction == "LEFT" else -AVOID_TURN_SPEED
            # No depende de un numero de pasos: espera a ver espacio realmente libre.
            if forward_corridor_clear(scan) and front_distance >= FRONT_CLEAR_DISTANCE:
                mode = MODE_AVOID_FOLLOW
                target_omega = 0.0
                target_speed = AVOID_FORWARD_SPEED
                print("FRENTE LIBRE: SIGUIENDO BORDE DEL OBSTACULO")
        elif mode == MODE_AVOID_FOLLOW:
            # Velocidad y giro ajustados continuamente por distancia al obstaculo.
            if not forward_corridor_clear(scan):
                mode = MODE_AVOID_TURN
                target_speed = 0.0
                target_omega = AVOID_TURN_SPEED if avoid_direction == "LEFT" else -AVOID_TURN_SPEED
            else:
                target_speed = AVOID_FORWARD_SPEED
                target_omega = follow_side_omega(avoid_direction, left_distance, right_distance, goal_angle)
                traveled = float(np.linalg.norm(planar_position(robot_node) - avoid_start_position))
                if traveled >= AVOID_MIN_TRAVEL and abs(goal_angle) < AVOID_HEADING_TOLERANCE and forward_corridor_clear(scan) and front_distance > FRONT_CLEAR_DISTANCE:
                    mode = MODE_GOAL
                    print("EVASION TERMINADA: RETOMANDO {}".format(current_destination_name))
        if navigation_active and mode in (MODE_AVOID_TURN, MODE_AVOID_FOLLOW) and robot.getTime() - avoid_started > AVOID_MAX_SECONDS:
            print("EVASION SIN SALIDA EN {:.0f} s: STOP".format(AVOID_MAX_SECONDS))
            if registro is not None:
                registro.cancelar_por("evasion_sin_salida")
            navigation_active = False
            current_destination = None
            current_destination_name = None
            mode = MODE_GOAL
            target_speed = 0.0
            target_omega = 0.0
        if navigation_active and step_counter % LOG_EVERY_STEPS == 0:
            print("{} | {} | Dist {:.2f} m | Ang {:.2f} rad | {}".format(current_destination_name, mode, distance, goal_angle, scan_text(scan)))

    # SEGURIDAD FINAL: no permitir avance si la huella frontal esta ocupada.
    parada_seguridad = False
    if not lidar_ok:
        target_speed = 0.0
        target_omega = 0.0
    if target_speed > 0.0 and not forward_corridor_clear(scan):
        target_speed = 0.0
        target_omega = 0.0
        parada_seguridad = True
        if step_counter % LOG_EVERY_STEPS == 0:
            print("SEGURIDAD CORREDOR: {} | STOP".format(scan_text(scan)))
    # Precaucion con rotaciones junto a maquinaria cercana.
    if abs(target_omega) > 0.0 and target_speed == 0.0:
        turn_direction = "LEFT" if target_omega > 0 else "RIGHT"
        if not turn_clearance(scan, turn_direction):
            target_omega = 0.0
            if step_counter % LOG_EVERY_STEPS == 0:
                print("GIRO BLOQUEADO: {} | {}".format(turn_direction, scan_text(scan)))
    if target_speed < 0.0 and back_distance < SAFE_DISTANCE_BACK:
        target_speed = 0.0
    if mode == MODE_WAIT_PERSON or person_detected:
        target_speed = 0.0
        target_omega = 0.0

    # TURN_SIGN corrige la direccion fisica de los motores si fuera necesario.
    target_omega *= TURN_SIGN
    left_speed = (target_speed - target_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
    right_speed = (target_speed + target_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
    motor_left_wheel.setVelocity(left_speed)
    motor_right_wheel.setVelocity(right_speed)

    # Experimento: medir el paso, detectar el final y cerrar la simulacion
    if registro is not None and registro.iniciado and not registro.terminado:
        ruta_node = destination_nodes[ord(str(EXPERIMENTO["ruta"]))]
        registro.paso(planar_position(robot_node), scan, mode, time_step / 1000.0, parada_seguridad)
        if not navigation_active:
            registro.terminar("cancelada", robot.getTime(), goal_geometry(robot_node, ruta_node)[0])
        elif robot.getTime() - registro.t_inicio_sim > EXPERIMENTO["max_sim_s"]:
            registro.terminar("tiempo_agotado", robot.getTime(), goal_geometry(robot_node, ruta_node)[0])
    if registro is not None and registro.terminado and not salida_enviada:
        salida_enviada = True
        motor_left_wheel.setVelocity(0.0)
        motor_right_wheel.setVelocity(0.0)
        robot.simulationQuit(0)

motor_left_wheel.setVelocity(0.0)
motor_right_wheel.setVelocity(0.0)
cv2.destroyAllWindows()
