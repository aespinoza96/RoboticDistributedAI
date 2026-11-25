from controller import Supervisor, Keyboard
from ultralytics import YOLO
import numpy as np
import cv2
import math
import os


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

from puente import PuenteComandos  # noqa: E402  (mismo directorio del controlador)
from lidar import distancias_por_sector  # noqa: E402
import lidar  # noqa: E402
import profundidad  # noqa: E402
import mapa as mapa_mod  # noqa: E402
import planificador  # noqa: E402

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
AVOID_MAX_SECONDS = 25.0          # No insistir indefinidamente ante una ruta bloqueada.
AVOID_HEADING_TOLERANCE = 0.30
MAX_AVOID_ATTEMPTS = 8

YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.55
YOLO_EVERY_N_STEPS = 3
PERSON_CLEAR_DETECTIONS = 5       # 5 inferencias seguidas sin persona antes de reanudar.
LOG_EVERY_STEPS = 30

# Camara de profundidad: obstaculos por encima del plano del LiDAR (mesas, cintas).
PROFUNDIDAD_ACTIVA = os.getenv("MIR_PROFUNDIDAD", "1") != "0"   # MIR_PROFUNDIDAD=0 la apaga (comparaciones A/B)
PROFUNDIDAD_CADA_N_PASOS = 2       # se procesa cada N pasos; la ultima lectura se reutiliza
OBSTACULO_ALTO_AVISO_SEGUNDOS = 10.0

SAFE_DISTANCE_SIDE = 0.60      # Margen inicial conservador; calibrar para el robot.
SAFE_DISTANCE_DIAGONAL = 0.95  # No atravesar esquinas de maquinaria.
TURN_CLEARANCE_MIN = 0.65      # Distancia lateral minima para permitir una rotacion.
LIDAR_UNKNOWN_IS_BLOCKED = True
LOG_SECTOR_INTERVAL = 30

# Puente de comandos (agente / MCP): TCP local, JSON por linea. Ver puente.py.
PUENTE_ACTIVO = True
PUENTE_PUERTO = int(os.getenv("MIR_PUENTE_PUERTO", "5555"))

# Vigilancia de tareas: convierte bloqueos silenciosos en fallos con motivo.
SIN_PROGRESO_SEGUNDOS = 20.0       # Navegando sin moverse SIN_PROGRESO_DISTANCIA en este tiempo = fallo.
BLOQUEO_NAVEGACION_SEGUNDOS = 5.0  # La seguridad frenando la navegacion este tiempo = fallo con la regla exacta.
PUNTO_MAX_COORDENADA = 50.0        # ir_a_punto: limite de cordura para x e y (m).
SIN_PROGRESO_DISTANCIA = 0.10
LIDAR_SIN_DATOS_SEGUNDOS = 10.0    # LiDAR sin lecturas suficientes durante una tarea = fallo.
PERSONA_BLOQUEANDO_SEGUNDOS = 15.0 # Aviso (no fallo) si una persona frena la tarea tanto tiempo.

# Maniobras cortas pedidas por el agente; la capa de seguridad sigue aplicando.
MANIOBRA_MAX_GRADOS = 180.0
MANIOBRA_MAX_METROS = 1.0
MANIOBRA_VELOCIDAD = 0.12
MANIOBRA_GIRO = 0.25
MANIOBRA_TIEMPO_MAX = 30.0
MANIOBRA_BLOQUEO_SEGUNDOS = 3.0    # Seguridad bloqueando la maniobra este tiempo = fallo.

# Mapa de ocupacion + ruta A* (planificador.py): ir_a / ir_a_punto planifican la ruta
# alrededor de lo conocido en vez de ir en linea recta y esquivar a ciegas.
PLANIFICADOR_ACTIVO = os.getenv("MIR_PLANIFICADOR", "1") != "0"   # MIR_PLANIFICADOR=0 vuelve a la navegacion reactiva
MAPA_CADA_N_PASOS = 3
PLAN_VELOCIDAD = 0.30
PLAN_OMEGA = 0.30
# Siguiendo una ruta planificada (que ya guarda distancia a lo conocido), la seguridad
# final usa margenes de choque, no de comodidad: si no, frena rutas validas en pasillos.
PLAN_FRENTE_MIN = 0.35
PLAN_DIAGONAL_MIN = 0.25
PLAN_LADO_MIN = 0.12
PLAN_GIRO_MIN = 0.25
PLAN_LLEGADA_CERCA = 0.60          # la meta cae en un obstaculo: llegar a esto cuenta como exito

MODE_PLAN = "PLAN"
MODE_GOAL = "GOAL"
MODE_AVOID_TURN = "AVOID_TURN"
MODE_AVOID_FOLLOW = "AVOID_FOLLOW"
MODE_WAIT_PERSON = "WAIT_PERSON"

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


def goal_geometry(robot_node, target):
    # target: punto en el plano del suelo (np.array de 3, como planar_position).
    displacement = target - planar_position(robot_node)
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


def read_lidar_distances(front_lidar, back_lidar):
    # Sectores por angulo real en el marco del robot (ver lidar.py); las distancias
    # son holgura al borde de la huella del robot. Una zona sin rayos validos
    # nunca se considera despejada: su sector vale 0.0 y ok=False.
    empty = {"left": 0.0, "front_left": 0.0, "front": 0.0, "front_right": 0.0, "right": 0.0, "back": 0.0, "ok": False}
    try:
        front_values = front_lidar.getLayerRangeImage(0)
        back_values = back_lidar.getLayerRangeImage(0)
        if not front_values or not back_values:
            return empty
        data = distancias_por_sector(front_values, back_values, front_lidar.getMaxRange(), back_lidar.getMaxRange(),
                                     math.degrees(front_lidar.getFov()))
    except Exception as error:
        print("ERROR LIDAR: {}".format(error))
        return empty

    if LIDAR_SWAP_SIDES:
        data["left"], data["right"] = data["right"], data["left"]
        data["front_left"], data["front_right"] = data["front_right"], data["front_left"]

    data["ok"] = all(data[k] > 0.0 for k in ("left", "front_left", "front", "front_right", "right", "back"))
    return data


def forward_corridor_clear(scan):
    return scan["ok"] and scan["front"] >= SAFE_DISTANCE_FRONT and scan["front_left"] >= SAFE_DISTANCE_DIAGONAL and scan["front_right"] >= SAFE_DISTANCE_DIAGONAL and scan["left"] >= SAFE_DISTANCE_SIDE and scan["right"] >= SAFE_DISTANCE_SIDE


def plan_forward_clear(scan):
    return scan["ok"] and scan["front"] >= PLAN_FRENTE_MIN and scan["front_left"] >= PLAN_DIAGONAL_MIN \
        and scan["front_right"] >= PLAN_DIAGONAL_MIN and scan["left"] >= PLAN_LADO_MIN and scan["right"] >= PLAN_LADO_MIN


def plan_turn_clear(scan, direction):
    if not scan["ok"]:
        return False
    if direction == "LEFT":
        return scan["left"] >= PLAN_GIRO_MIN and scan["front_left"] >= PLAN_GIRO_MIN
    return scan["right"] >= PLAN_GIRO_MIN and scan["front_right"] >= PLAN_GIRO_MIN


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


def choose_avoid_direction(scan):
    # Regla local: solo lados con espacio para girar (costado Y diagonal);
    # entre ellos, el que tenga mas espacio libre. None si ninguno sirve.
    candidates = []
    for direction, side, diagonal in (("LEFT", "left", "front_left"), ("RIGHT", "right", "front_right")):
        if turn_clearance(scan, direction) and scan[side] >= SIDE_MIN_DISTANCE:
            candidates.append((min(scan[side], scan[diagonal]), direction))
    if not candidates:
        return None
    return max(candidates)[1]


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


def planar_forward(node):
    # Vector unitario del frente del robot proyectado en el plano del suelo.
    up = axis_vector("+" + WORLD_UP_AXIS)
    orientation = np.array(node.getOrientation(), dtype=float).reshape((3, 3))
    forward = orientation.dot(axis_vector(ROBOT_FORWARD_AXIS))
    forward -= np.dot(forward, up) * up
    norm = float(np.linalg.norm(forward))
    return forward / norm if norm > 0.0001 else forward


def heading_delta(previous_forward, forward):
    # Angulo con signo (rad, + = izquierda) entre dos orientaciones consecutivas.
    up = axis_vector("+" + WORLD_UP_AXIS)
    return math.atan2(float(np.dot(up, np.cross(previous_forward, forward))), float(np.dot(previous_forward, forward)))


def robot_pose(node):
    # (x, y, yaw rad) para el mapa; el planificador asume Z hacia arriba.
    position = planar_position(node)
    return float(position[0]), float(position[1]), math.radians(robot_yaw_degrees(node))


def robot_yaw_degrees(node):
    forward = planar_forward(node)
    if WORLD_UP_AXIS == "Z":
        return math.degrees(math.atan2(forward[1], forward[0]))
    if WORLD_UP_AXIS == "Y":
        return math.degrees(math.atan2(-forward[2], forward[0]))
    return math.degrees(math.atan2(forward[0], forward[1]))


def normalize_destination(value, names):
    # Acepta "ENTREGA_3", "entrega 3", "Entrega-3", "3" o 3.
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    if text in names:
        return text
    if text.isdigit() and "ENTREGA_{}".format(text) in names:
        return "ENTREGA_{}".format(text)
    return None


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

depth_camera = robot.getDevice("depth_camera") if PROFUNDIDAD_ACTIVA else None
if depth_camera is not None:
    depth_camera.enable(time_step * PROFUNDIDAD_CADA_N_PASOS)
    print("CAMARA DE PROFUNDIDAD: {}x{}, {:.0f} grados, {:.2f}-{:.2f} m".format(
        depth_camera.getWidth(), depth_camera.getHeight(), math.degrees(depth_camera.getFov()),
        depth_camera.getMinRange(), depth_camera.getMaxRange()))
elif PROFUNDIDAD_ACTIVA:
    print("ADVERTENCIA: no hay 'depth_camera'; solo LiDAR (no ve obstaculos por encima de su plano).")
depth_sectors = {}
depth_hits = []
depth_points_xy = (np.zeros(0), np.zeros(0))
occupancy_map = mapa_mod.Mapa()
navigator = planificador.Navegador(occupancy_map, PLAN_VELOCIDAD, PLAN_OMEGA)
if PLANIFICADOR_ACTIVO and WORLD_UP_AXIS != "Z":
    print("ADVERTENCIA: el planificador asume Z hacia arriba; se usa la navegacion reactiva.")
    PLANIFICADOR_ACTIVO = False
depth_warned_at = -1e9
depth_error_printed = False


def read_depth_image(device):
    # getRangeImage(data_type="buffer") devuelve un puntero ctypes a floats: se lee
    # sin convertir 921.600 valores a una lista de Python.
    pointer = device.getRangeImage(data_type="buffer")
    if not pointer:
        return None
    height, width = device.getHeight(), device.getWidth()
    return np.ctypeslib.as_array(pointer, shape=(height * width,)).reshape((height, width)).copy()

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
scan = {"left": 0.0, "front_left": 0.0, "front": 0.0, "front_right": 0.0, "right": 0.0, "back": 0.0, "ok": False}

# Tareas: cada orden (teclado o puente) es una tarea con un final explicito:
# estado "exito", "fallo" o "cancelada", y un motivo legible por el agente.
task = None
last_task = None
task_counter = 0
maneuver = None
safety_block = None
progress_position = None
progress_time = 0.0
lidar_bad_since = None
nav_blocked_since = None
person_since = None
person_warned = False

destination_by_name = {DESTINATIONS[key]: node for key, node in destination_nodes.items()}

bridge = None
if PUENTE_ACTIVO:
    try:
        bridge = PuenteComandos(puerto=PUENTE_PUERTO)
        print("PUENTE DE COMANDOS: escuchando en 127.0.0.1:{}".format(bridge.puerto))
    except OSError as error:
        print("ADVERTENCIA: no se pudo abrir el puente en el puerto {} ({}). Solo teclado.".format(PUENTE_PUERTO, error))


def emit(event, data=None):
    if bridge is not None:
        bridge.emitir(event, robot.getTime(), data)


def robot_context():
    position = planar_position(robot_node)
    context = {
        "posicion": [round(float(position[0]), 2), round(float(position[1]), 2)],
        "orientacion_grados": round(robot_yaw_degrees(robot_node), 1),
        "modo": mode,
        "navegando": navigation_active,
        "destino": current_destination_name,
        "lidar": {k: round(scan[k], 2) for k in ("left", "front_left", "front", "front_right", "right", "back")},
        "lidar_ok": scan["ok"],
        "persona_detectada": person_detected,
        "obstaculos_altos": depth_hits,
        "mapa_conocido_m2": known_map_m2(),
    }
    if navigation_active and mode == MODE_PLAN and navigator.ruta:
        remaining = navigator.metros_restantes(robot_pose(robot_node))
        context["ruta_planificada_m"] = round(remaining, 2) if remaining is not None else None
    if current_destination is not None:
        context["distancia_destino"] = round(goal_geometry(robot_node, current_destination)[0], 2)
    return context


def start_task(kind, args, origin):
    global task, task_counter
    if task is not None:
        finish_task("cancelada", "reemplazada", "Una nueva orden ({}) reemplazo a esta tarea".format(kind))
    task_counter += 1
    task = {"tarea_id": task_counter, "tipo": kind, "args": args, "origen": origin, "estado": "en_curso", "inicio": round(robot.getTime(), 2)}
    emit("tarea_iniciada", dict(task))
    return task_counter


def finish_task(state, reason, detail=""):
    # Cierra la tarea en curso y avisa a los clientes con el contexto completo.
    global task, last_task
    if task is None:
        return
    task.update({"estado": state, "motivo": reason, "detalle": detail, "fin": round(robot.getTime(), 2), "contexto": robot_context()})
    last_task = task
    task = None
    print("TAREA {} {}: {} | {}".format(last_task["tarea_id"], state.upper(), reason, detail))
    emit("tarea_terminada", dict(last_task))


def stop_navigation():
    global navigation_active, current_destination, current_destination_name, mode, manual_speed, manual_omega
    navigator.detener()
    navigation_active = False
    current_destination = None
    current_destination_name = None
    mode = MODE_GOAL
    manual_speed = 0.0
    manual_omega = 0.0


def start_navigation(name, target=None):
    # name: DEF de una entrega, o etiqueta de un punto (target dado en coordenadas).
    global navigation_active, current_destination, current_destination_name, mode, saved_mode
    global avoid_attempts, manual_speed, manual_omega, maneuver, progress_position, progress_time, lidar_bad_since
    global nav_blocked_since
    maneuver = None
    lidar_bad_since = None
    nav_blocked_since = None
    current_destination = planar_position(destination_by_name[name]) if target is None else target
    current_destination_name = name
    navigation_active = True
    base_mode = MODE_PLAN if PLANIFICADOR_ACTIVO else MODE_GOAL
    mode = MODE_WAIT_PERSON if person_detected else base_mode
    saved_mode = base_mode
    if PLANIFICADOR_ACTIVO:
        navigator.iniciar((float(current_destination[0]), float(current_destination[1])))
    avoid_attempts = 0
    manual_speed = 0.0
    manual_omega = 0.0
    progress_position = planar_position(robot_node)
    progress_time = robot.getTime()
    print("\nINICIANDO RUTA: {}".format(name))


def route_length(target):
    """Metros de ruta A* desde la pose actual hasta target (x, y) con el mapa actual, o None."""
    probe = planificador.Navegador(occupancy_map, PLAN_VELOCIDAD, PLAN_OMEGA)
    probe.iniciar(target)
    pose = robot_pose(robot_node)
    cells = probe.planificar(pose)
    if cells is None:
        return None
    points = [(pose[0], pose[1])] + [occupancy_map.centro(f, c) for f, c in cells]
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def known_map_m2():
    return round(float(np.count_nonzero(occupancy_map.conocido())) * occupancy_map.resolucion ** 2, 1)


def start_maneuver(kind, amount):
    global maneuver, manual_speed, manual_omega
    stop_navigation()
    manual_speed = 0.0
    manual_omega = 0.0
    maneuver = {"tipo": kind, "objetivo": amount, "hecho": 0.0, "inicio": robot.getTime(),
                "posicion_inicial": planar_position(robot_node), "frente_previo": planar_forward(robot_node),
                "bloqueado_desde": None}


def handle_command(client, message):
    global maneuver
    cmd = message["cmd"]
    args = message.get("args", {})
    request_id = message.get("id")

    def ok(result=None):
        bridge.responder(client, request_id, True, result)

    def fail(error):
        bridge.responder(client, request_id, False, error=error)

    try:
        if cmd == "estado":
            ok({"contexto": robot_context(), "tarea": task, "ultima_tarea": last_task,
                "maniobra": None if maneuver is None else {"tipo": maneuver["tipo"], "objetivo": maneuver["objetivo"], "hecho": round(maneuver["hecho"], 3)}})
        elif cmd == "destinos":
            here = planar_position(robot_node)
            ok({"destinos": [{"nombre": name, "posicion": [round(float(p[0]), 2), round(float(p[1]), 2)],
                              "distancia": round(float(np.linalg.norm(p - here)), 2)}
                             for name, p in ((n, planar_position(node)) for n, node in destination_by_name.items())]})
        elif cmd == "escanear":
            ok({"lidar": {k: round(scan[k], 2) for k in ("left", "front_left", "front", "front_right", "right", "back")},
                "lidar_ok": scan["ok"], "frente_libre": forward_corridor_clear(scan),
                "puede_girar_izquierda": turn_clearance(scan, "LEFT"), "puede_girar_derecha": turn_clearance(scan, "RIGHT"),
                "lado_sugerido_para_evadir": choose_avoid_direction(scan), "persona_detectada": person_detected,
                "obstaculos_altos": depth_hits,
                "nota": "Holgura en metros desde el borde del robot; 0.0 = sin lecturas validas (se trata como bloqueado). "
                        "Combina LiDAR (plano a 0.19 m) y camara de profundidad (al frente, hasta 1.95 m); "
                        "obstaculos_altos = sectores donde la profundidad vio algo mas cerca que el LiDAR (p. ej. una mesa)."})
        elif cmd == "volcar_mapa":
            # Diagnostico: guarda las capas del mapa (.npz) para ver que bloquea las rutas.
            path = str(args.get("ruta", ""))
            if not path.endswith(".npz"):
                fail("'ruta' debe terminar en .npz")
                return
            np.savez(path, log_odds=occupancy_map.log_odds, alto=occupancy_map.alto, ocupado=occupancy_map.ocupado(),
                     origen=np.array([occupancy_map.x_min, occupancy_map.y_min]), resolucion=occupancy_map.resolucion)
            ok({"ruta": path, "contexto": robot_context()})
        elif cmd == "volcar_profundidad":
            # Diagnostico: guarda la imagen de profundidad cruda (.npy) para calibrar.
            if depth_camera is None:
                fail("No hay camara de profundidad")
                return
            path = str(args.get("ruta", ""))
            if not path.endswith(".npy"):
                fail("'ruta' debe terminar en .npy")
                return
            image = read_depth_image(depth_camera)
            if image is None:
                fail("La camara de profundidad aun no tiene imagen")
                return
            np.save(path, image)
            ok({"ruta": path, "contexto": robot_context()})
        elif cmd == "ir_a":
            name = normalize_destination(args.get("destino", ""), destination_by_name)
            if name is None:
                fail("Destino desconocido '{}'. Disponibles: {}".format(args.get("destino"), ", ".join(sorted(destination_by_name))))
                return
            task_id = start_task("ir_a", {"destino": name}, "puente")
            start_navigation(name)
            ok({"tarea_id": task_id, "destino": name})
        elif cmd == "ruta":
            # Distancia por ruta (A* sobre el mapa actual) hasta un destino o un punto, sin moverse.
            if "destino" in args:
                name = normalize_destination(args.get("destino", ""), destination_by_name)
                if name is None:
                    fail("Destino desconocido '{}'. Disponibles: {}".format(args.get("destino"), ", ".join(sorted(destination_by_name))))
                    return
                goal = planar_position(destination_by_name[name])
                label = name
            else:
                goal = np.array([float(args.get("x")), float(args.get("y")), 0.0])
                label = "PUNTO({:.2f}, {:.2f})".format(goal[0], goal[1])
            here = planar_position(robot_node)
            meters = route_length((float(goal[0]), float(goal[1])))
            ok({"destino": label, "hay_ruta": meters is not None,
                "metros_por_ruta": None if meters is None else round(meters, 2),
                "metros_en_linea_recta": round(float(np.linalg.norm(goal - here)), 2),
                "mapa_conocido_m2": known_map_m2()})
        elif cmd == "ir_a_punto":
            # Punto intermedio (waypoint) en coordenadas del mundo, para rodear obstaculos.
            x, y = float(args.get("x")), float(args.get("y"))
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > PUNTO_MAX_COORDENADA or abs(y) > PUNTO_MAX_COORDENADA:
                fail("'x' e 'y' deben ser numeros finitos de como maximo {} en valor absoluto".format(PUNTO_MAX_COORDENADA))
                return
            up = axis_vector("+" + WORLD_UP_AXIS)
            target = np.array([x, y, 0.0]) if WORLD_UP_AXIS == "Z" else np.array([x, 0.0, y])
            target = target - np.dot(target, up) * up
            label = "PUNTO({:.2f}, {:.2f})".format(x, y)
            task_id = start_task("ir_a_punto", {"x": x, "y": y}, "puente")
            start_navigation(label, target)
            ok({"tarea_id": task_id, "destino": label})
        elif cmd in ("girar", "avanzar"):
            key, limit = ("grados", MANIOBRA_MAX_GRADOS) if cmd == "girar" else ("metros", MANIOBRA_MAX_METROS)
            amount = float(args.get(key, 0.0))
            if amount == 0.0 or abs(amount) > limit:
                fail("'{}' debe ser distinto de 0 y como maximo {} en valor absoluto".format(key, limit))
                return
            task_id = start_task(cmd, {key: amount}, "puente")
            start_maneuver(cmd, math.radians(amount) if cmd == "girar" else amount)
            ok({"tarea_id": task_id, key: amount})
        elif cmd == "parar":
            had_task = task is not None
            finish_task("cancelada", "parar", "Cancelada por orden 'parar'")
            stop_navigation()
            maneuver = None
            ok({"detenido": True, "habia_tarea": had_task})
        else:
            fail("Comando desconocido '{}'. Comandos: estado, destinos, escanear, ruta, ir_a, ir_a_punto, girar, avanzar, parar, volcar_profundidad".format(cmd))
    except (TypeError, ValueError) as error:
        fail("Argumentos invalidos para '{}': {}".format(cmd, error))


print("\nMiR100: YOLO + LiDAR + NAVEGACION LOCAL REACTIVA")
print("Teclas 1/2/3: destino | 0 o ESPACIO: cancelar | Flechas: control manual")
print("Evasion: regla local con LiDAR (lado con mas espacio libre)")
print("PRECAUCION: prueba inicial en simulacion despejada, velocidad baja\n")

# ===================================================
# BUCLE PRINCIPAL
# ===================================================
while robot.step(time_step) != -1:
    step_counter += 1
    scan = read_lidar_distances(front_lidar, back_lidar)
    if depth_camera is not None:
        if step_counter % PROFUNDIDAD_CADA_N_PASOS == 0:
            try:
                depth_image = read_depth_image(depth_camera)
                if depth_image is not None:
                    px, py, pz = profundidad.puntos_obstaculo_xyz(
                        depth_image, depth_camera.getFov(), depth_camera.getMinRange(), depth_camera.getMaxRange())
                    depth_points_xy = (px, py)
                    depth_sectors = profundidad.holgura_por_sector(*profundidad.holguras_de_puntos(px, py, pz))
            except Exception as error:
                depth_sectors = {}
                if not depth_error_printed:
                    depth_error_printed = True
                    print("ERROR PROFUNDIDAD (se sigue solo con LiDAR): {}".format(error))
        scan, depth_hits = profundidad.combinar(scan, depth_sectors)
        if depth_hits and task is not None and robot.getTime() - depth_warned_at > OBSTACULO_ALTO_AVISO_SEGUNDOS:
            depth_warned_at = robot.getTime()
            emit("obstaculo_alto", {"tarea_id": task["tarea_id"], "sectores": depth_hits,
                                    "detalle": "La camara de profundidad ve un obstaculo por encima del plano del LiDAR (por ejemplo una mesa)."})
    # El mapa se construye siempre (tambien sin tarea): es la memoria del entorno.
    if step_counter % MAPA_CADA_N_PASOS == 0:
        try:
            pose = robot_pose(robot_node)
            for lidar_device, first_angle, position in ((front_lidar, lidar.PRIMER_ANGULO_FRONTAL, lidar.POSICION_FRONTAL),
                                                        (back_lidar, lidar.PRIMER_ANGULO_TRASERO, lidar.POSICION_TRASERO)):
                occupancy_map.actualizar_lidar(pose, lidar_device.getLayerRangeImage(0), first_angle, position,
                                               lidar_device.getMaxRange(), math.degrees(lidar_device.getFov()))
            occupancy_map.actualizar_alto(pose, *depth_points_xy)
        except Exception as error:
            print("ERROR MAPA: {}".format(error))
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
        cv2.imshow("MiR100 - YOLO todas las clases", annotated_frame if annotated_frame is not None else frame)
        cv2.waitKey(1)

    # Leer todas las teclas; ultimo evento tiene prioridad.
    key = keyboard.getKey()
    while True:
        next_key = keyboard.getKey()
        if next_key == -1:
            break
        key = next_key

    if key in destination_nodes:
        start_task("ir_a", {"destino": DESTINATIONS[key]}, "teclado")
        start_navigation(DESTINATIONS[key])
    elif key == ord("0") or key == ord(" "):
        finish_task("cancelada", "teclado", "Cancelada con la tecla 0/Espacio")
        stop_navigation()
        maneuver = None
        print("NAVEGACION CANCELADA: STOP")
    elif not navigation_active and maneuver is None:
        if key == Keyboard.UP:
            manual_speed = min(SPEED_MAX, manual_speed + SPEED_INCREMENT)
        elif key == Keyboard.DOWN:
            manual_speed = max(SPEED_MIN, manual_speed - SPEED_INCREMENT)
        elif key == Keyboard.LEFT:
            manual_omega = min(ANGULAR_SPEED_MAX, manual_omega + ANGULAR_SPEED_INCREMENT)
        elif key == Keyboard.RIGHT:
            manual_omega = max(-ANGULAR_SPEED_MAX, manual_omega - ANGULAR_SPEED_INCREMENT)

    # Ordenes del puente (agente / MCP). Se procesan aqui, en el hilo de Webots.
    if bridge is not None:
        for client, message in bridge.poll():
            handle_command(client, message)

    now = robot.getTime()
    target_speed = manual_speed if not navigation_active and maneuver is None else 0.0
    target_omega = manual_omega if not navigation_active and maneuver is None else 0.0

    # Persona: se conservan los datos de evasion, se frena y se reanuda despues.
    if person_detected and mode != MODE_WAIT_PERSON:
        saved_mode = mode
        mode = MODE_WAIT_PERSON
        manual_speed = 0.0
        manual_omega = 0.0
        person_since = now
        person_warned = False
        print("PERSONA DETECTADA (confianza {:.2f}): STOP".format(person_confidence))
        if task is not None:
            emit("persona_detectada", {"tarea_id": task["tarea_id"], "confianza": round(person_confidence, 2)})

    if mode == MODE_WAIT_PERSON:
        target_speed = 0.0
        target_omega = 0.0
        # No cuenta como falta de progreso ni consume el tiempo de una maniobra.
        progress_position = planar_position(robot_node)
        progress_time = now
        if maneuver is not None:
            maneuver["inicio"] += time_step / 1000.0
        if task is not None and not person_warned and person_since is not None and now - person_since > PERSONA_BLOQUEANDO_SEGUNDOS:
            person_warned = True
            emit("persona_bloqueando", {"tarea_id": task["tarea_id"], "segundos": round(now - person_since, 1),
                                        "detalle": "Una persona frena la tarea; el robot sigue esperando. Se puede esperar mas o cancelar."})
        if not person_detected and person_clear_count >= PERSON_CLEAR_DETECTIONS:
            mode = saved_mode
            if mode in (MODE_AVOID_TURN, MODE_AVOID_FOLLOW):
                avoid_started = now
            print("PERSONA DESPEJADA: REANUDANDO")
            if task is not None:
                emit("persona_despejada", {"tarea_id": task["tarea_id"]})

    elif navigation_active and current_destination is not None:
        distance, goal_angle = goal_geometry(robot_node, current_destination)
        if lidar_ok:
            lidar_bad_since = None
        if distance <= ARRIVAL_DISTANCE:
            print("\nDESTINO ALCANZADO: {} | Distancia {:.2f} m".format(current_destination_name, distance))
            finish_task("exito", "llego", "Llego a {} (a {:.2f} m)".format(current_destination_name, distance))
            stop_navigation()
            target_speed = 0.0
            target_omega = 0.0
        elif not lidar_ok:
            target_speed = 0.0
            target_omega = 0.0
            if lidar_bad_since is None:
                lidar_bad_since = now
                emit("lidar_sin_datos", {"tarea_id": task["tarea_id"] if task else None, "lidar": {k: round(scan[k], 2) for k in ("left", "front_left", "front", "front_right", "right", "back")}})
            elif now - lidar_bad_since > LIDAR_SIN_DATOS_SEGUNDOS:
                finish_task("fallo", "lidar_sin_datos", "El LiDAR no da lecturas suficientes desde hace {:.0f} s; sin datos se considera bloqueado".format(now - lidar_bad_since))
                stop_navigation()
            if step_counter % LOG_EVERY_STEPS == 0:
                print("SEGURIDAD: LiDAR sin lecturas suficientes, STOP")
        elif mode == MODE_PLAN:
            target_speed, target_omega, plan_result = navigator.paso(robot_pose(robot_node), now)
            if plan_result == planificador.LLEGO:
                if distance <= PLAN_LLEGADA_CERCA:
                    print("\nDESTINO ALCANZADO (ruta): {} | Distancia {:.2f} m".format(current_destination_name, distance))
                    finish_task("exito", "llego", "Llego a {} por ruta planificada (a {:.2f} m)".format(current_destination_name, distance))
                else:
                    finish_task("fallo", "meta_inalcanzable", "La meta {} esta dentro de un obstaculo segun el mapa; el robot llego lo mas cerca posible (a {:.2f} m)".format(
                        current_destination_name, distance))
                stop_navigation()
                target_speed = target_omega = 0.0
            elif plan_result == planificador.SIN_RUTA:
                print("SIN RUTA A {}".format(current_destination_name))
                finish_task("fallo", "sin_ruta", "El mapa no tiene ningun camino libre hacia {} (distancia {:.2f} m): todo lo conocido alrededor esta bloqueado | {}".format(
                    current_destination_name, distance, scan_text(scan)))
                stop_navigation()
                target_speed = target_omega = 0.0
        elif mode == MODE_GOAL:
            # Girar hacia destino sin avanzar hasta que este orientado.
            if abs(goal_angle) > NAVIGATION_ANGLE_TOLERANCE:
                target_speed = 0.0
                target_omega = clamp(goal_angle * 1.2, -NAVIGATION_TURN_SPEED, NAVIGATION_TURN_SPEED)
            elif forward_corridor_clear(scan):
                target_speed = min(NAVIGATION_SPEED, max(0.06, 0.5 * distance))
                target_omega = clamp(goal_angle * 0.5, -NAVIGATION_TURN_SPEED, NAVIGATION_TURN_SPEED)
            else:
                # Obstruccion: la regla local elige el lado; la capa de seguridad sigue al final.
                target_speed = 0.0
                target_omega = 0.0
                direction = choose_avoid_direction(scan)
                if avoid_attempts >= MAX_AVOID_ATTEMPTS:
                    print("LIMITE DE EVASIONES: STOP")
                    finish_task("fallo", "limite_evasiones", "Obstaculo al frente tras {} evasiones; distancia al destino {:.2f} m | {}".format(avoid_attempts, distance, scan_text(scan)))
                    stop_navigation()
                elif direction is None:
                    print("NO HAY ESPACIO LATERAL SUFICIENTE: STOP")
                    finish_task("fallo", "sin_espacio_lateral", "Obstaculo al frente y ningun lado con espacio para girar | {}".format(scan_text(scan)))
                    stop_navigation()
                else:
                    avoid_direction = direction
                    avoid_attempts += 1
                    mode = MODE_AVOID_TURN
                    avoid_started = robot.getTime()
                    avoid_start_position = planar_position(robot_node)
                    print("OBSTACULO: GIRAR {} | {} | destino {}".format(avoid_direction, scan_text(scan), current_destination_name))
                    emit("evasion_iniciada", {"tarea_id": task["tarea_id"] if task else None, "direccion": avoid_direction,
                                              "intento": avoid_attempts, "max_intentos": MAX_AVOID_ATTEMPTS, "lidar": scan_text(scan)})
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
                    emit("evasion_terminada", {"tarea_id": task["tarea_id"] if task else None, "distancia_destino": round(distance, 2)})
        if navigation_active and mode in (MODE_AVOID_TURN, MODE_AVOID_FOLLOW) and robot.getTime() - avoid_started > AVOID_MAX_SECONDS:
            print("EVASION SIN SALIDA EN {:.0f} s: STOP".format(AVOID_MAX_SECONDS))
            finish_task("fallo", "evasion_sin_salida", "La evasion hacia {} no encontro salida en {:.0f} s | {}".format(avoid_direction, AVOID_MAX_SECONDS, scan_text(scan)))
            stop_navigation()
            target_speed = 0.0
            target_omega = 0.0
        # Vigilancia: navegando sin desplazarse = atascado (aunque ninguna regla haya saltado).
        if navigation_active:
            here = planar_position(robot_node)
            if float(np.linalg.norm(here - progress_position)) >= SIN_PROGRESO_DISTANCIA:
                progress_position = here
                progress_time = now
            elif now - progress_time > SIN_PROGRESO_SEGUNDOS:
                print("SIN PROGRESO EN {:.0f} s: STOP".format(SIN_PROGRESO_SEGUNDOS))
                finish_task("fallo", "sin_progreso", "Navegando sin avanzar {:.2f} m en {:.0f} s (modo {}); distancia al destino {:.2f} m | {}".format(
                    SIN_PROGRESO_DISTANCIA, SIN_PROGRESO_SEGUNDOS, mode, distance, scan_text(scan)))
                stop_navigation()
                target_speed = 0.0
                target_omega = 0.0
        if navigation_active and step_counter % LOG_EVERY_STEPS == 0:
            print("{} | {} | Dist {:.2f} m | Ang {:.2f} rad | {}".format(current_destination_name, mode, distance, goal_angle, scan_text(scan)))

    elif maneuver is not None:
        # Maniobra corta pedida por el agente (girar N grados / avanzar N metros).
        elapsed = now - maneuver["inicio"]
        if maneuver["tipo"] == "girar":
            forward = planar_forward(robot_node)
            maneuver["hecho"] += heading_delta(maneuver["frente_previo"], forward)
            maneuver["frente_previo"] = forward
            remaining = maneuver["objetivo"] - maneuver["hecho"]
            done = abs(remaining) < math.radians(2.0)
            target_omega = 0.0 if done else math.copysign(clamp(abs(remaining) * 1.5, 0.05, MANIOBRA_GIRO), remaining)
        else:
            maneuver["hecho"] = float(np.linalg.norm(planar_position(robot_node) - maneuver["posicion_inicial"]))
            remaining = abs(maneuver["objetivo"]) - maneuver["hecho"]
            done = remaining < 0.03
            target_speed = 0.0 if done else math.copysign(clamp(remaining * 0.8, 0.05, MANIOBRA_VELOCIDAD), maneuver["objetivo"])
        if done:
            amount = math.degrees(maneuver["hecho"]) if maneuver["tipo"] == "girar" else maneuver["hecho"]
            finish_task("exito", "maniobra_completa", "{} completado: {:.2f} {}".format(maneuver["tipo"], amount, "grados" if maneuver["tipo"] == "girar" else "m"))
            maneuver = None
            target_speed = 0.0
            target_omega = 0.0
        elif elapsed > MANIOBRA_TIEMPO_MAX:
            finish_task("fallo", "tiempo_agotado", "La maniobra {} no termino en {:.0f} s".format(maneuver["tipo"], MANIOBRA_TIEMPO_MAX))
            maneuver = None
            target_speed = 0.0
            target_omega = 0.0

    # SEGURIDAD FINAL: no permitir avance si la huella frontal esta ocupada.
    requested_speed, requested_omega = target_speed, target_omega
    safety_block = None
    if not lidar_ok:
        target_speed = 0.0
        target_omega = 0.0
        safety_block = "lidar_sin_datos"
    planned = navigation_active and mode == MODE_PLAN
    if target_speed > 0.0 and not (plan_forward_clear(scan) if planned else forward_corridor_clear(scan)):
        target_speed = 0.0
        if not planned:
            target_omega = 0.0  # en ruta planificada se puede seguir girando hacia la ruta
        safety_block = "corredor_frontal_ocupado"
        if planned:
            navigator.replanificar_ya()
        if step_counter % LOG_EVERY_STEPS == 0:
            print("SEGURIDAD CORREDOR: {} | STOP".format(scan_text(scan)))
    # Precaucion con rotaciones junto a maquinaria cercana.
    if abs(target_omega) > 0.0 and target_speed == 0.0:
        turn_direction = "LEFT" if target_omega > 0 else "RIGHT"
        if not (plan_turn_clear(scan, turn_direction) if planned else turn_clearance(scan, turn_direction)):
            target_omega = 0.0
            safety_block = "sin_espacio_para_girar_" + ("izquierda" if turn_direction == "LEFT" else "derecha")
            if step_counter % LOG_EVERY_STEPS == 0:
                print("GIRO BLOQUEADO: {} | {}".format(turn_direction, scan_text(scan)))
    if target_speed < 0.0 and back_distance < SAFE_DISTANCE_BACK:
        target_speed = 0.0
        safety_block = "obstaculo_atras"
    if mode == MODE_WAIT_PERSON or person_detected:
        target_speed = 0.0
        target_omega = 0.0
        safety_block = "persona"

    # Navegacion frenada por la seguridad (por ejemplo, no puede girar hacia el destino):
    # se informa en segundos con la regla exacta, en vez de esperar al vigilante de progreso.
    if not navigation_active or mode == MODE_WAIT_PERSON or person_detected:
        nav_blocked_since = None  # esperar a una persona no es un bloqueo
    else:
        nav_blocked = (requested_speed != 0.0 and target_speed == 0.0) or (requested_omega != 0.0 and target_omega == 0.0)
        if not nav_blocked:
            nav_blocked_since = None
        elif nav_blocked_since is None:
            nav_blocked_since = now
        elif now - nav_blocked_since > BLOQUEO_NAVEGACION_SEGUNDOS:
            print("NAVEGACION BLOQUEADA POR SEGURIDAD ({}): STOP".format(safety_block))
            finish_task("fallo", "bloqueado_por_seguridad", "La seguridad impide seguir hacia {} desde hace {:.0f} s ({}, modo {}) | {} | atras {:.2f}".format(
                current_destination_name, now - nav_blocked_since, safety_block, mode, scan_text(scan), back_distance))
            stop_navigation()
            target_speed = 0.0
            target_omega = 0.0

    # Una maniobra que la seguridad frena durante un rato falla con el motivo exacto.
    if maneuver is not None and mode != MODE_WAIT_PERSON:
        blocked = (requested_speed != 0.0 and target_speed == 0.0) or (requested_omega != 0.0 and target_omega == 0.0)
        if not blocked:
            maneuver["bloqueado_desde"] = None
        elif maneuver["bloqueado_desde"] is None:
            maneuver["bloqueado_desde"] = now
        elif now - maneuver["bloqueado_desde"] > MANIOBRA_BLOQUEO_SEGUNDOS:
            finish_task("fallo", "maniobra_bloqueada", "La seguridad impide {} ({}) | {} | atras {:.2f}".format(
                maneuver["tipo"], safety_block, scan_text(scan), back_distance))
            maneuver = None
            target_speed = 0.0
            target_omega = 0.0

    # TURN_SIGN corrige la direccion fisica de los motores si fuera necesario.
    target_omega *= TURN_SIGN
    left_speed = (target_speed - target_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
    right_speed = (target_speed + target_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
    motor_left_wheel.setVelocity(left_speed)
    motor_right_wheel.setVelocity(right_speed)

motor_left_wheel.setVelocity(0.0)
motor_right_wheel.setVelocity(0.0)
if bridge is not None:
    bridge.cerrar()
cv2.destroyAllWindows()
