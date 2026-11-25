# MiR100 - Webots: memoria espacial real con LiDAR, A*, YOLO y Jev opcional.
# MODO 1: pulsa M y recorre manualmente los pasillos para aprenderlos.
# MODO 2: pulsa 1, 2 o 3 para planificar por espacio realmente observado.
# Z es vertical; XY es el suelo. Cambiar orientaciones del LiDAR solo despues de calibrarlas.
from controller import Supervisor, Keyboard
from ultralytics import YOLO
from memoria_espacial import MemoriaEspacial
import numpy as np
import cv2
import math
import os
import time
import threading
import queue

try:
    from typesafe_sdk import TypeSafeClient, Choice
    JEV_AVAILABLE = True
except ImportError:
    JEV_AVAILABLE = False

WHEEL_RADIUS = 0.0625
DISTANCE_TO_CENTER = 0.2226
NAVIGATION_SPEED = 0.16
MANUAL_SPEED_MAX = 0.22
REVERSE_SPEED_MAX = 0.10
ANGULAR_SPEED_MAX = 0.25
SAFE_FRONT = 0.85
SAFE_DIAGONAL = 0.72
SAFE_SIDE = 0.52
SAFE_REAR = 0.80
SAFE_ROTATION = 0.64
# Geometria conservadora del MiR100; ajustar tras validar dimensiones en Webots.
FOOTPRINT_RADIUS = 0.46
FOOTPRINT_MARGIN = 0.10
ROTATION_RADIUS = 0.59
PATH_LOOKAHEAD = 0.88
REVERSE_LOOKAHEAD = 0.65
LIDAR_MIN_VALID_FRACTION = 0.65
MAP_VIEW_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factory_map_debug.png')
ARRIVAL_DISTANCE = 0.40
WAYPOINT_DISTANCE = 0.28
MAP_RESOLUTION = 0.20
MAP_ROBOT_RADIUS = 0.52  # Radio conservador + margen: ajustar a la huella real.
MAP_BOUNDS = (-9.0, -9.0, 9.0, 9.0)
MAPPING_EVERY_STEPS = 6
PLANNING_MIN_INTERVAL = 2.0
REPLAN_MAX_ATTEMPTS = 4
REPLAN_BLOCKED_STEPS = 22
TEMP_BLOCK_RADIUS = 0.28
TEMP_BLOCK_SECONDS = 30.0
REPLAN_MIN_PROGRESS = 0.18
REPLAN_OBSTACLE_SAME_DISTANCE = 0.22
PATH_START_MAX_DISTANCE = 0.55
TURN_BLOCKED_MAX_STEPS = 20
# Recuperacion de giro bloqueado: retroceso corto SOLO con el LiDAR valido y pasillo posterior despejado.
RECOVERY_MAX_ATTEMPTS = 2
RECOVERY_REVERSE_DISTANCE = 0.28
RECOVERY_REVERSE_SPEED = 0.075
RECOVERY_MAX_SECONDS = 7.0
RECOVERY_REAR_MIN = 1.00
RECOVERY_FORWARD_MIN = 1.00
RECOVERY_FORWARD_DISTANCE = 0.22
RECOVERY_REQUIRED_MOVEMENT = 0.07
RECOVERY_CLEAR_TURN_STEPS = 4
ROUTE_LOG_MAX_POINTS = 12
FINAL_APPROACH_DISTANCE = 0.85
SAVE_EVERY_SIM_SECONDS = 25.0
YOLO_CONFIDENCE = 0.55
YOLO_EVERY_STEPS = 4
PERSON_CLEAR_DETECTIONS = 5
DESTINATIONS = {ord('1'): 'ENTREGA_1', ord('2'): 'ENTREGA_2', ord('3'): 'ENTREGA_3'}
ROBOT_FORWARD_AXIS = '+X'
TURN_SIGN = 1.0
LIDAR_SWAP_SIDES = False
# Eje del primer rayo del LiDAR: Webots suele escanear de izquierda a derecha.
# Comprobar en Show Lidar Rays Paths antes de confiar en el mapa.
LIDAR_LOCAL_FORWARD = '+X'
LIDAR_LOCAL_LEFT = '+Y'
LIDAR_ANGLE_SIGN = -1.0
LIDAR_CENTER_OFFSET = 0.0
JEV_ENABLED = True
JEV_REQUEST_TIMEOUT = 8.0
MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factory_map.json')


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def unit_axis(name):
    axes = {'+X': [1.0, 0.0, 0.0], '-X': [-1.0, 0.0, 0.0], '+Y': [0.0, 1.0, 0.0], '-Y': [0.0, -1.0, 0.0], '+Z': [0.0, 0.0, 1.0], '-Z': [0.0, 0.0, -1.0]}
    return np.array(axes[name], dtype=float)


def heading(node):
    orientation = np.array(node.getOrientation(), dtype=float).reshape((3, 3))
    direction = orientation.dot(unit_axis(ROBOT_FORWARD_AXIS))
    return math.atan2(float(direction[1]), float(direction[0]))


def waypoint_command(robot_node, waypoint, speed):
    position = robot_node.getPosition()
    dx, dy = waypoint[0] - position[0], waypoint[1] - position[1]
    distance = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    error = math.atan2(math.sin(target_angle - heading(robot_node)), math.cos(target_angle - heading(robot_node)))
    if abs(error) > 0.16:
        return 0.0, clamp(error * 0.9, -ANGULAR_SPEED_MAX, ANGULAR_SPEED_MAX), distance
    return min(speed, max(0.06, distance * 0.6)), clamp(error * 0.5, -0.12, 0.12), distance


def valid_ranges(lidar):
    # Inf = rayo sin retorno, que permite mapear espacio libre solo hasta maxRange.
    try:
        raw = np.asarray(lidar.getLayerRangeImage(lidar.getNumberOfLayers() // 2), dtype=float)
        maximum = float(lidar.getMaxRange())
        if len(raw) == 0 or maximum <= 0.0:
            return None, 0.0
        raw[np.isposinf(raw)] = maximum
        raw[~np.isfinite(raw)] = 0.0
        raw[(raw < 0.05) | (raw > maximum)] = 0.0
        return raw, maximum
    except Exception as error:
        print('ERROR al leer LiDAR: {}'.format(error))
        return None, 0.0


def sector(values, a, b, maximum):
    if values is None:
        return 0.0
    total = len(values)
    part = values[int(total * a):max(int(total * b), int(total * a) + 1)]
    good = part[part > 0.0]
    if len(good) < max(1, int(len(part) * 0.45)):
        return 0.0
    return float(np.percentile(good, 15))


def read_scan(front, back):
    values, fmax = valid_ranges(front)
    rear, rmax = valid_ranges(back)
    scan = {'I': sector(values, 0.06, 0.23, fmax), 'DI': sector(values, 0.23, 0.41, fmax), 'F': sector(values, 0.41, 0.59, fmax), 'DD': sector(values, 0.59, 0.77, fmax), 'D': sector(values, 0.77, 0.94, fmax), 'B': sector(rear, 0.35, 0.65, rmax)}
    if LIDAR_SWAP_SIDES:
        scan['I'], scan['D'] = scan['D'], scan['I']
        scan['DI'], scan['DD'] = scan['DD'], scan['DI']
    scan['ok'] = min(scan.values()) > 0.0
    return scan


def lidar_hits(lidar, sensor_node):
    """Retornos reales del LiDAR transformados a XY global; no depende de sectores resumidos."""
    values, maximum = valid_ranges(lidar)
    if values is None or sensor_node is None:
        return [], False
    valid = int(np.count_nonzero(values > 0.0))
    if valid < max(10, int(len(values) * LIDAR_MIN_VALID_FRACTION)):
        return [], False
    rotation = np.array(sensor_node.getOrientation(), dtype=float).reshape((3, 3))
    origin = sensor_node.getPosition()
    forward = unit_axis(LIDAR_LOCAL_FORWARD)
    left = unit_axis(LIDAR_LOCAL_LEFT)
    count = len(values)
    fov = float(lidar.getFov())
    points = []
    for index in range(count):
        distance = float(values[index])
        if not (0.05 < distance < maximum - 0.08):
            continue
        angle = LIDAR_CENTER_OFFSET + LIDAR_ANGLE_SIGN * (index / float(max(1, count - 1)) - 0.5) * fov
        local = math.cos(angle) * forward + math.sin(angle) * left
        direction = rotation.dot(local)
        if abs(float(direction[2])) > 0.24:
            continue
        points.append((float(origin[0] + distance * direction[0]), float(origin[1] + distance * direction[1])))
    return points, True


def build_environment(front, front_node, back, back_node):
    front_points, front_ok = lidar_hits(front, front_node)
    rear_points, back_ok = lidar_hits(back, back_node)
    # Cada punto conserva su sensor de origen, para diagnosticar la envolvente de giro.
    tagged = [(point, 'FRONTAL') for point in front_points] + [(point, 'TRASERO') for point in rear_points]
    return {'points': front_points + rear_points, 'tagged': tagged, 'ok': front_ok and back_ok, 'front_count': len(front_points), 'back_count': len(rear_points)}


def blocking_hit_diagnostic(robot_node, environment, point, scan):
    if point is None:
        return 'Sin retorno LiDAR identificable'
    pose = robot_node.getPosition()
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    distance = math.hypot(dx, dy)
    azimuth = math.atan2(dy, dx)
    local_angle = math.atan2(math.sin(azimuth - heading(robot_node)), math.cos(azimuth - heading(robot_node)))
    nearest_sensor = 'NO IDENTIFICADO'
    best = float('inf')
    for hit, sensor in environment.get('tagged', []):
        error = math.hypot(hit[0] - point[0], hit[1] - point[1])
        if error < best:
            best, nearest_sensor = error, sensor
    return 'Sensor {} | obstaculo X {:.2f} Y {:.2f} | al centro {:.2f} m | angulo local {:.2f} rad | B {:.2f} m | radio giro {:.2f} m'.format(nearest_sensor, point[0], point[1], distance, local_angle, scan['B'], ROTATION_RADIUS)


def can_begin_recovery(robot_node, environment, scan, person):
    # Probar primero retroceso y, si no es viable, avance corto. Ninguna opcion fuerza una colision.
    if person or not environment['ok'] or not scan['ok']:
        return 0, 'PERSONA DETECTADA O LIDAR INVALIDO'
    if scan['B'] >= RECOVERY_REAR_MIN:
        permitted, point, reason = clearance_for_motion(robot_node, environment, -RECOVERY_REVERSE_SPEED, 0.0)
        if permitted:
            return -1, 'RETROCESO'
    if scan['F'] >= RECOVERY_FORWARD_MIN:
        permitted, point, reason = clearance_for_motion(robot_node, environment, RECOVERY_REVERSE_SPEED, 0.0)
        if permitted:
            return 1, 'AVANCE'
    return 0, 'SIN PASILLO SEGURO PARA REPOSICIONAR'


def recovery_motion_clear(robot_node, environment, direction, person):
    if person or not environment['ok']:
        return False, 'PERSONA DETECTADA O LIDAR INVALIDO'
    permitted, point, reason = clearance_for_motion(robot_node, environment, direction * RECOVERY_REVERSE_SPEED, 0.0)
    return permitted, reason


def clearance_for_motion(robot_node, environment, speed, omega):
    """Valida barrido recto o rotacion en sitio. Ante lecturas invalidas se detiene."""
    if not environment['ok']:
        return False, None, 'LIDAR_NO_DISPONIBLE'
    position = robot_node.getPosition()
    yaw = heading(robot_node)
    fx, fy = math.cos(yaw), math.sin(yaw)
    lx, ly = -fy, fx
    radius = FOOTPRINT_RADIUS + FOOTPRINT_MARGIN
    near = None
    near_distance = float('inf')
    turning = abs(omega) > 0.01 and abs(speed) < 0.015
    for point in environment['points']:
        dx, dy = point[0] - position[0], point[1] - position[1]
        distance = math.hypot(dx, dy)
        if turning:
            blocked = distance <= ROTATION_RADIUS
        elif speed > 0.0:
            longitudinal = dx * fx + dy * fy
            lateral = dx * lx + dy * ly
            blocked = -0.08 <= longitudinal <= PATH_LOOKAHEAD and abs(lateral) <= radius
        elif speed < 0.0:
            longitudinal = -(dx * fx + dy * fy)
            lateral = dx * lx + dy * ly
            blocked = -0.08 <= longitudinal <= REVERSE_LOOKAHEAD and abs(lateral) <= radius
        else:
            blocked = False
        if blocked and distance < near_distance:
            near, near_distance = point, distance
    if near is not None:
        return False, near, 'BARRIDO_GIRO_BLOQUEADO' if turning else 'CORREDOR_BLOQUEADO'
    return True, None, 'DESPEJADO'


def safe_drive(speed, omega, environment, robot_node, person):
    if person:
        return 0.0, 0.0, None, 'PERSONA_DETECTADA'
    if abs(speed) > 0.01 and abs(omega) > 0.13:
        return 0.0, 0.0, None, 'CURVATURA_NO_VALIDADA'
    permitted, obstacle, reason = clearance_for_motion(robot_node, environment, speed, omega)
    if not permitted:
        return 0.0, 0.0, obstacle, reason
    return speed, omega, None, 'OK'


def show_map(memoria, robot_node, route_points, destination_name):
    """Mapa debug en PNG: ocupacion, trayectoria y pose en coordenadas correctas."""
    height, width = memoria.height, memoria.width
    frame = np.full((height, width, 3), 125, dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            status = memoria.cell_status((c, r))
            frame[height - 1 - r, c] = (228, 236, 225) if status == 'LIBRE' else ((60, 72, 85) if status == 'OCUPADA' else ((30, 90, 235) if status == 'BLOQUEO_TEMPORAL' else (165, 165, 165)))
    def pixel(x, y):
        cell = memoria.world_to_cell(x, y)
        return (cell[0], height - 1 - cell[1]) if cell is not None else None
    for i in range(len(route_points) - 1):
        a, b = pixel(*route_points[i]), pixel(*route_points[i + 1])
        if a is not None and b is not None:
            cv2.line(frame, a, b, (15, 180, 20), 1)
    pose = robot_node.getPosition()
    robot_pixel = pixel(pose[0], pose[1])
    if robot_pixel:
        cv2.circle(frame, robot_pixel, 2, (230, 80, 25), -1)
        aim = pixel(pose[0] + 0.60 * math.cos(heading(robot_node)), pose[1] + 0.60 * math.sin(heading(robot_node)))
        if aim:
            cv2.line(frame, robot_pixel, aim, (230, 80, 25), 1)
    scale = 5
    frame = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(frame, 'Mapa MiR100 | {}'.format(destination_name or 'Exploracion'), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    cv2.imwrite(MAP_VIEW_FILE, frame)
    print('MAPA VISUAL GUARDADO: {}'.format(MAP_VIEW_FILE))


def map_lidar(memoria, lidar, sensor_node):
    # Las poses de los sensores se obtienen por Supervisor.getFromDevice().
    values, maximum = valid_ranges(lidar)
    if values is None or sensor_node is None:
        return False
    rotation = np.array(sensor_node.getOrientation(), dtype=float).reshape((3, 3))
    origin = sensor_node.getPosition()
    basis_forward = unit_axis(LIDAR_LOCAL_FORWARD)
    basis_left = unit_axis(LIDAR_LOCAL_LEFT)
    fov = float(lidar.getFov())
    count = len(values)
    # Submuestreo de rayos para no ralentizar demasiado Webots.
    stride = max(1, int(count / 110))
    for i in range(0, count, stride):
        reading = float(values[i])
        if reading <= 0.0:
            continue
        ray_angle = LIDAR_CENTER_OFFSET + LIDAR_ANGLE_SIGN * ((i / float(max(1, count - 1))) - 0.5) * fov
        local = math.cos(ray_angle) * basis_forward + math.sin(ray_angle) * basis_left
        world = rotation.dot(local)
        if abs(world[2]) > 0.24:
            continue  # Ignorar rayos demasiado inclinados en el mapa horizontal.
        azimuth = math.atan2(float(world[1]), float(world[0]))
        hit = reading < maximum - 0.08
        memoria.observe_ray((origin[0], origin[1]), azimuth, reading, maximum, hit)
    return True


def block_detected_front(memoria, robot_node, lidar, lidar_node, now):
    """Marca el eco frontal más cercano en XY; evita inventar obstáculos sin un retorno LiDAR."""
    values, maximum = valid_ranges(lidar)
    if values is None or lidar_node is None:
        return None
    count = len(values)
    start = int(count * 0.41)
    end = max(start + 1, int(count * 0.59))
    candidates = [(float(values[i]), i) for i in range(start, end) if 0.05 < values[i] < maximum - 0.08]
    if not candidates:
        return None
    distance, i = min(candidates)
    rotation = np.array(lidar_node.getOrientation(), dtype=float).reshape((3, 3))
    origin = lidar_node.getPosition()
    angle = LIDAR_CENTER_OFFSET + LIDAR_ANGLE_SIGN * ((i / float(max(1, count - 1))) - 0.5) * float(lidar.getFov())
    local = math.cos(angle) * unit_axis(LIDAR_LOCAL_FORWARD) + math.sin(angle) * unit_axis(LIDAR_LOCAL_LEFT)
    world = rotation.dot(local)
    if abs(world[2]) > 0.24:
        return None
    x = float(origin[0] + distance * world[0])
    y = float(origin[1] + distance * world[1])
    robot_xy = robot_node.getPosition()
    if math.hypot(x - robot_xy[0], y - robot_xy[1]) < 0.25:
        return None
    memoria.mark_temporary_obstacle(x, y, radius=TEMP_BLOCK_RADIUS, expires_at=now + TEMP_BLOCK_SECONDS)
    return (x, y, distance)


def print_map_diagnostic(memoria, robot_node, obstacle, route_points=None):
    position = robot_node.getPosition()
    cell = memoria.world_to_cell(obstacle[0], obstacle[1])
    status = memoria.cell_status(cell)
    print('DIAGNOSTICO MAPA | Robot X {:.2f} Y {:.2f} | Orientacion {:.2f} rad'.format(position[0], position[1], heading(robot_node)))
    print('DIAGNOSTICO MAPA | Obstaculo X {:.2f} Y {:.2f} | Celda {} | Estado {}'.format(obstacle[0], obstacle[1], cell, status))
    if route_points:
        print('RUTA ANTERIOR | {} puntos'.format(len(route_points)))
        for index, point in enumerate(route_points[:ROUTE_LOG_MAX_POINTS]):
            print('  PUNTO {} | X {:.2f} Y {:.2f}'.format(index + 1, point[0], point[1]))


def validate_candidate(memoria, start_xy, candidate):
    if not candidate:
        return False, 'sin ruta'
    first = candidate[0]
    if math.hypot(first[0] - start_xy[0], first[1] - start_xy[1]) > PATH_START_MAX_DISTANCE:
        return False, 'el inicio de A* esta demasiado lejos del robot (posible celda inicial inflada)'
    if len(candidate) > 1 and not memoria.segment_is_clear(start_xy, candidate[1], allow_start_unknown=False):
        return False, 'el primer tramo no atraviesa solo celdas libres'
    return True, 'OK'


def route_signature(points, resolution=0.20):
    """Resumen de trayectoria para detectar replanteamientos idénticos."""
    return tuple((int(round(p[0] / resolution)), int(round(p[1] / resolution))) for p in points)


def jev_worker(ticket, observation, results):
    try:
        with TypeSafeClient(model='jev-latest') as client:
            response = client.system_one(state=observation, questions={'decision': Choice(instructions='You are advising a mapping robot that has stopped because no verified collision-free route is currently available. Choose whether more mapping is needed or stopping is best. Do not command movement.', criteria={'MAP_MORE': 'The robot should gather more sensor data by supervised manual mapping', 'STOP': 'Stay stopped because the geometry is uncertain or blocked'})})
        answer = response.choices['decision']
        results.put((ticket, str(answer.choice).upper(), float(answer.confidence), None))
    except Exception as error:
        results.put((ticket, None, 0.0, str(error)))


robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()
if robot_node is None:
    raise RuntimeError('Activa supervisor TRUE en el nodo principal del MiR100')
left_motor = robot.getDevice('middle_left_wheel_joint')
right_motor = robot.getDevice('middle_right_wheel_joint')
for motor in (left_motor, right_motor):
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
front_lidar = robot.getDevice('front_lidar')
back_lidar = robot.getDevice('back_lidar')
front_lidar.enable(time_step)
back_lidar.enable(time_step)
front_node = robot.getFromDevice(front_lidar._tag)
back_node = robot.getFromDevice(back_lidar._tag)
if front_node is None or back_node is None:
    print('AVISO: No se pudieron localizar los nodos LiDAR: no se guardara un mapa incorrecto.')

camera = robot.getDevice('camera')
camera.enable(time_step)
keyboard = Keyboard()
keyboard.enable(time_step)
model = YOLO('yolo11n.pt')
destinations = {}
for key, name in DESTINATIONS.items():
    node = robot.getFromDef(name)
    if node is not None:
        destinations[key] = node
        print('DESTINO: {} -> {}'.format(name, node.getPosition()))
    else:
        print('DESTINO NO ENCONTRADO: {}'.format(name))

if os.path.isfile(MAP_FILE):
    try:
        memoria = MemoriaEspacial.load(MAP_FILE)
        print('MEMORIA ESPACIAL CARGADA: {}'.format(MAP_FILE))
    except Exception as error:
        print('MAPA INCOMPATIBLE: {}. Creando mapa nuevo.'.format(error))
        memoria = MemoriaEspacial(*MAP_BOUNDS, resolution=MAP_RESOLUTION, robot_radius=MAP_ROBOT_RADIUS)
else:
    memoria = MemoriaEspacial(*MAP_BOUNDS, resolution=MAP_RESOLUTION, robot_radius=MAP_ROBOT_RADIUS)

mapping_active = True
navigation_active = False
current_goal = None
current_name = None
route = []
route_index = 0
blocked_steps = 0
replan_attempts = 0
last_route_signature = None
last_route_start_position = None
last_replan_time = -999.0
last_block_obstacle = None
last_block_robot_position = None
turn_blocked_steps = 0
last_motion_obstacle = None
last_motion_reason = 'OK'
last_motion_blocked = False
recovery_active = False
recovery_start_xy = None
recovery_direction = 0
recovery_start_time = 0.0
recovery_attempts = 0
recovery_turn_clear_count = 0
last_progress_position = None
last_progress_time = robot.getTime()
last_save_time = robot.getTime()
step_counter = 0
manual_speed = 0.0
manual_omega = 0.0
person_detected = False
person_clear_count = 0
annotated = None
jev_results = queue.Queue()
jev_ticket = 0
jev_pending = False
jev_requested_at = 0.0

print('\nMiR100 MEMORIA ESPACIAL + A* + YOLO + JEV')
print('M: activar/desactivar mapeo | S: guardar mapa | P: estadisticas | V: exportar PNG mapa')
print('FLECHAS: explorar manualmente | 1/2/3: navegar usando mapa | 0/ESPACIO: cancelar')
print('RECUPERACION DE GIRO: maximo {} intentos de retroceso <= {:.2f} m, siempre validando el LiDAR trasero'.format(RECOVERY_MAX_ATTEMPTS, RECOVERY_REVERSE_DISTANCE))
print('IMPORTANTE: primero mapear los pasillos manualmente. No se asume que lo desconocido esta libre.\n')

try:
    while robot.step(time_step) != -1:
        step_counter += 1
        scan = read_scan(front_lidar, back_lidar)
        environment = build_environment(front_lidar, front_node, back_lidar, back_node)
        memoria.expire_temporary_obstacles(robot.getTime())
        if mapping_active and step_counter % MAPPING_EVERY_STEPS == 0:
            map_lidar(memoria, front_lidar, front_node)
            map_lidar(memoria, back_lidar, back_node)

        image = camera.getImage()
        if image:
            frame = np.frombuffer(image, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if step_counter % YOLO_EVERY_STEPS == 0:
                results = model.predict(frame, conf=YOLO_CONFIDENCE, verbose=False)
                person_detected = False
                if results and results[0].boxes is not None:
                    person_detected = any(int(b.cls[0]) == 0 for b in results[0].boxes)
                person_clear_count = 0 if person_detected else person_clear_count + 1
                annotated = results[0].plot() if results else frame
            cv2.imshow('MiR100 YOLO', annotated if annotated is not None else frame)
            cv2.waitKey(1)

        keys = []
        while True:
            key = keyboard.getKey()
            if key == -1:
                break
            keys.append(key)
        for key in keys:
            if key == ord('M') or key == ord('m'):
                mapping_active = not mapping_active
                print('MAPEO: {}'.format('ACTIVO' if mapping_active else 'PAUSADO'))
            elif key == ord('S') or key == ord('s'):
                memoria.save(MAP_FILE)
                print('MAPA GUARDADO: {}'.format(MAP_FILE))
            elif key == ord('P') or key == ord('p'):
                print('MAPA | libres {}, ocupadas {}, desconocidas {}'.format(*memoria.stats()))
            elif key == ord('V') or key == ord('v'):
                show_map(memoria, robot_node, route, current_name)
            elif key == ord('0') or key == ord(' '):
                navigation_active = False
                route = []
                route_index = 0
                recovery_active = False
                recovery_attempts = 0
                manual_speed = 0.0
                manual_omega = 0.0
                jev_ticket += 1
                jev_pending = False
                print('STOP: navegacion cancelada')
            elif key in destinations:
                navigation_active = True
                current_goal = destinations[key]
                current_name = DESTINATIONS[key]
                route = []
                route_index = 0
                blocked_steps = 0
                replan_attempts = 0
                last_route_signature = None
                last_route_start_position = None
                last_replan_time = -999.0
                last_block_obstacle = None
                last_block_robot_position = None
                turn_blocked_steps = 0
                recovery_active = False
                recovery_attempts = 0
                recovery_start_xy = None
                recovery_turn_clear_count = 0
                last_motion_obstacle = None
                last_motion_reason = 'OK'
                last_progress_position = None
                last_progress_time = robot.getTime()
                manual_speed = 0.0
                manual_omega = 0.0
                jev_ticket += 1
                jev_pending = False
                print('PLANIFICANDO HACIA {}'.format(current_name))
            elif not navigation_active and key == Keyboard.UP:
                manual_speed = clamp(manual_speed + 0.05, -REVERSE_SPEED_MAX, MANUAL_SPEED_MAX)
            elif not navigation_active and key == Keyboard.DOWN:
                manual_speed = clamp(manual_speed - 0.05, -REVERSE_SPEED_MAX, MANUAL_SPEED_MAX)
            elif not navigation_active and key == Keyboard.LEFT:
                manual_omega = clamp(manual_omega + 0.05, -ANGULAR_SPEED_MAX, ANGULAR_SPEED_MAX)
            elif not navigation_active and key == Keyboard.RIGHT:
                manual_omega = clamp(manual_omega - 0.05, -ANGULAR_SPEED_MAX, ANGULAR_SPEED_MAX)

        while not jev_results.empty():
            ticket, action, confidence, error = jev_results.get_nowait()
            if ticket == jev_ticket:
                jev_pending = False
                if error:
                    print('JEV ERROR: {}'.format(error))
                else:
                    print('JEV ASESOR: {} | confianza {:.2f}'.format(action, confidence))

        if jev_pending and time.monotonic() - jev_requested_at > JEV_REQUEST_TIMEOUT:
            jev_ticket += 1
            jev_pending = False
            print('JEV TIMEOUT (no afecta la seguridad)')

        target_speed = manual_speed if not navigation_active else 0.0
        target_omega = manual_omega if not navigation_active else 0.0
        if navigation_active and current_goal is not None:
            position = robot_node.getPosition()
            goal = current_goal.getPosition()
            distance_to_goal = math.hypot(goal[0] - position[0], goal[1] - position[1])
            if distance_to_goal <= ARRIVAL_DISTANCE:
                print('DESTINO ALCANZADO: {} | {:.2f} m'.format(current_name, distance_to_goal))
                navigation_active = False
                route = []
                route_index = 0
                recovery_active = False
                target_speed = 0.0
                target_omega = 0.0
            else:
                if not route and robot.getTime() - last_replan_time >= PLANNING_MIN_INTERVAL:
                    last_replan_time = robot.getTime()
                    candidate = memoria.find_path((position[0], position[1]), (goal[0], goal[1]))
                    signature = route_signature(candidate, MAP_RESOLUTION)
                    valid_candidate, candidate_reason = validate_candidate(memoria, (position[0], position[1]), candidate)
                    if not candidate:
                        print('A*: SIN RUTA CONOCIDA hacia {}. Mapea otro pasillo.'.format(current_name))
                        navigation_active = False
                        route_index = 0
                        target_speed = 0.0
                        target_omega = 0.0
                        if JEV_ENABLED and JEV_AVAILABLE and os.getenv('TYPESAFE_API_KEY') and not jev_pending:
                            jev_ticket += 1
                            observation = {'destination': current_name, 'goal_distance_m': round(distance_to_goal, 2), 'lidar': {k: round(scan[k], 2) for k in ('I', 'DI', 'F', 'DD', 'D')}, 'map_status': 'No verified path over known free cells'}
                            threading.Thread(target=jev_worker, args=(jev_ticket, observation, jev_results), daemon=True).start()
                            jev_pending = True
                            jev_requested_at = time.monotonic()
                    elif not valid_candidate:
                        print('A*: RUTA DESCARTADA: {}'.format(candidate_reason))
                        for index, point in enumerate(candidate[:ROUTE_LOG_MAX_POINTS]):
                            print('  A* PUNTO {}: X {:.2f} Y {:.2f}'.format(index + 1, point[0], point[1]))
                        navigation_active = False
                        route = []
                        route_index = 0
                    elif last_route_signature is not None and signature == last_route_signature and last_route_start_position is not None and math.hypot(position[0] - last_route_start_position[0], position[1] - last_route_start_position[1]) < REPLAN_MIN_PROGRESS:
                        print('RUTA REPETIDA SIN AVANCE: STOP. Revisar obstaculo y mapa antes de continuar.')
                        navigation_active = False
                        route = []
                        route_index = 0
                        target_speed = 0.0
                        target_omega = 0.0
                    else:
                        route = candidate
                        route_index = 1 if len(route) > 1 else 0
                        last_route_signature = signature
                        last_route_start_position = (position[0], position[1])
                        print('A*: ruta a {} con {} puntos | intento {}'.format(current_name, len(route), replan_attempts))
                        if replan_attempts > 0:
                            for index, point in enumerate(route[:ROUTE_LOG_MAX_POINTS]):
                                print('  A* PUNTO {}: X {:.2f} Y {:.2f}'.format(index + 1, point[0], point[1]))
                if navigation_active and route:
                    if route_index >= len(route):
                        # Los puntos de A* pueden finalizar antes del centro exacto del destino.
                        # No borrar la ruta ni crear otra mientras quede una aproximacion corta.
                        if distance_to_goal <= FINAL_APPROACH_DISTANCE:
                            target_speed, target_omega, _ = waypoint_command(robot_node, (goal[0], goal[1]), NAVIGATION_SPEED * 0.65)
                        else:
                            print('ULTIMO PUNTO COMPLETADO, PERO DESTINO LEJANO: STOP ({:.2f} m)'.format(distance_to_goal))
                            navigation_active = False
                            target_speed = 0.0
                            target_omega = 0.0
                    else:
                        waypoint = route[route_index]
                        target_speed, target_omega, dist_wp = waypoint_command(robot_node, waypoint, NAVIGATION_SPEED)
                        if dist_wp <= WAYPOINT_DISTANCE:
                            route_index += 1
                            target_speed = 0.0
                            target_omega = 0.0
                            blocked_steps = 0
                    # Seguridad usa retornos reales del LiDAR en el corredor de movimiento.
                    # La recuperacion no fuerza giros: desplaza el robot solo si el corredor elegido esta despejado.
                    proposed_speed, proposed_omega = target_speed, target_omega
                    person_stop = person_detected or (person_clear_count < PERSON_CLEAR_DETECTIONS and step_counter > YOLO_EVERY_STEPS)
                    if recovery_active:
                        travelled_back = math.hypot(position[0] - recovery_start_xy[0], position[1] - recovery_start_xy[1])
                        elapsed = robot.getTime() - recovery_start_time
                        target_speed, target_omega = 0.0, 0.0
                        movement_clear, movement_reason = recovery_motion_clear(robot_node, environment, recovery_direction, person_stop)
                        if not movement_clear:
                            print('RECUPERACION CANCELADA: {} | {}'.format(movement_reason, blocking_hit_diagnostic(robot_node, environment, last_motion_obstacle, scan)))
                            recovery_active = False
                            navigation_active = False
                            route = []
                            route_index = 0
                        elif travelled_back >= (RECOVERY_REVERSE_DISTANCE if recovery_direction < 0 else RECOVERY_FORWARD_DISTANCE) or elapsed >= RECOVERY_MAX_SECONDS:
                            print('RECUPERACION FINALIZADA | movimiento {:.2f} m | tiempo {:.1f} s'.format(travelled_back, elapsed))
                            recovery_active = False
                            if travelled_back < RECOVERY_REQUIRED_MOVEMENT:
                                print('RECUPERACION SIN DESPLAZAMIENTO SUFICIENTE: STOP')
                                navigation_active = False
                                route = []
                                route_index = 0
                            else:
                                # Despues de reposicionar se calcula una ruta desde la nueva posicion.
                                route = []
                                route_index = 0
                                blocked_steps = 0
                                turn_blocked_steps = 0
                                last_route_signature = None
                                last_route_start_position = None
                                last_block_obstacle = None
                                last_block_robot_position = None
                                last_replan_time = robot.getTime() - PLANNING_MIN_INTERVAL
                                last_progress_position = (position[0], position[1])
                                last_progress_time = robot.getTime()
                                print('RECUPERACION: recalculando A* desde nueva posicion')
                        else:
                            target_speed = recovery_direction * RECOVERY_REVERSE_SPEED
                            target_omega = 0.0
                    else:
                        allowed, collision_point, collision_reason = clearance_for_motion(robot_node, environment, proposed_speed, proposed_omega)
                        moving = proposed_speed > 0.005
                        rotating = abs(proposed_omega) > 0.01 and abs(proposed_speed) < 0.015
                        if moving and not allowed:
                            blocked_steps += 1
                            target_speed, target_omega = 0.0, 0.0
                            last_motion_obstacle = collision_point
                            last_motion_reason = collision_reason
                        elif rotating and not allowed:
                            turn_blocked_steps += 1
                            target_speed, target_omega = 0.0, 0.0
                            last_motion_obstacle = collision_point
                            last_motion_reason = collision_reason
                            if turn_blocked_steps == 1 or turn_blocked_steps == TURN_BLOCKED_MAX_STEPS:
                                print('GIRO BLOQUEADO: {}'.format(blocking_hit_diagnostic(robot_node, environment, collision_point, scan)))
                        else:
                            if allowed and moving:
                                blocked_steps = 0
                                turn_blocked_steps = 0
                            if allowed and rotating:
                                turn_blocked_steps = 0
                        if navigation_active and turn_blocked_steps >= TURN_BLOCKED_MAX_STEPS:
                            print('GIRO IMPOSIBLE: envolvente de rotacion ocupada | intento recuperacion {}/{}'.format(recovery_attempts + 1, RECOVERY_MAX_ATTEMPTS))
                            recover_direction, recovery_reason = can_begin_recovery(robot_node, environment, scan, person_stop)
                            if recovery_attempts < RECOVERY_MAX_ATTEMPTS and recover_direction != 0:
                                recovery_attempts += 1
                                recovery_active = True
                                recovery_direction = recover_direction
                                recovery_start_xy = (position[0], position[1])
                                recovery_start_time = robot.getTime()
                                turn_blocked_steps = 0
                                blocked_steps = 0
                                target_speed, target_omega = 0.0, 0.0
                                print('RECUPERACION INICIADA | {} maximo {:.2f} m | F {:.2f} B {:.2f} | intento {}/{}'.format(recovery_reason, RECOVERY_REVERSE_DISTANCE if recovery_direction < 0 else RECOVERY_FORWARD_DISTANCE, scan['F'], scan['B'], recovery_attempts, RECOVERY_MAX_ATTEMPTS))
                            else:
                                print('GIRO SIN RECUPERACION SEGURA: {} | STOP'.format(recovery_reason if recover_direction == 0 else 'LIMITE DE INTENTOS'))
                                navigation_active = False
                                recovery_active = False
                                route = []
                                route_index = 0
                                target_speed, target_omega = 0.0, 0.0
                    if navigation_active and blocked_steps >= REPLAN_BLOCKED_STEPS:
                        blocked_steps = 0
                        replan_attempts += 1
                        obstacle = last_motion_obstacle
                        if obstacle is not None:
                            distancia_eco = math.hypot(obstacle[0] - position[0], obstacle[1] - position[1])
                            memoria.mark_temporary_obstacle(obstacle[0], obstacle[1], radius=TEMP_BLOCK_RADIUS, expires_at=robot.getTime() + TEMP_BLOCK_SECONDS)
                            print('OBSTACULO GEOLOCALIZADO | X {:.2f} Y {:.2f} | distancia al centro {:.2f} m | {}'.format(obstacle[0], obstacle[1], distancia_eco, last_motion_reason))
                            print_map_diagnostic(memoria, robot_node, (obstacle[0], obstacle[1], distancia_eco), route)
                            current_xy = (position[0], position[1])
                            repeated_obstacle = last_block_obstacle is not None and math.hypot(obstacle[0] - last_block_obstacle[0], obstacle[1] - last_block_obstacle[1]) <= REPLAN_OBSTACLE_SAME_DISTANCE
                            no_progress = last_block_robot_position is not None and math.hypot(current_xy[0] - last_block_robot_position[0], current_xy[1] - last_block_robot_position[1]) < REPLAN_MIN_PROGRESS
                            if repeated_obstacle and no_progress:
                                print('BLOQUEO PERSISTENTE: mismo punto de colision y sin progreso; detener')
                                navigation_active = False
                            last_block_obstacle = obstacle
                            last_block_robot_position = current_xy
                        else:
                            print('BLOQUEO SIN ECO LOCALIZABLE: STOP. Revisar calibracion LiDAR')
                            navigation_active = False
                        if replan_attempts >= REPLAN_MAX_ATTEMPTS:
                            print('LIMITE DE REPLANIFICACIONES: STOP')
                            navigation_active = False
                        if navigation_active:
                            route = []
                            route_index = 0
                            last_replan_time = robot.getTime() - PLANNING_MIN_INTERVAL
                            print('OBSTACULO NUEVO: A* recalcular con zona temporalmente bloqueada ({}/{})'.format(replan_attempts, REPLAN_MAX_ATTEMPTS))
                        else:
                            route = []
                            route_index = 0
                if navigation_active and not recovery_active and last_progress_position is not None:
                    travelled = math.hypot(position[0] - last_progress_position[0], position[1] - last_progress_position[1])
                    if travelled > 0.20:
                        last_progress_time = robot.getTime()
                        last_progress_position = (position[0], position[1])
                        if replan_attempts > 0:
                            replan_attempts = max(0, replan_attempts - 1)
                    elif robot.getTime() - last_progress_time > 15.0 and abs(target_speed) > 0.03:
                        print('SIN PROGRESO REAL por 15 s: STOP')
                        navigation_active = False
                        route = []
                        route_index = 0
                elif navigation_active and not recovery_active:
                    last_progress_position = (position[0], position[1])
                    last_progress_time = robot.getTime()
                if navigation_active and step_counter % 50 == 0:
                    total_tramos = max(0, len(route) - 1)
                    tramo_actual = min(route_index, total_tramos) if route else 0
                    print('{} | distancia {:.2f} m | tramo {}/{} | I {:.2f} DI {:.2f} F {:.2f} DD {:.2f} D {:.2f}'.format(current_name, distance_to_goal, tramo_actual, total_tramos, scan['I'], scan['DI'], scan['F'], scan['DD'], scan['D']))

        if recovery_active and not navigation_active:
            recovery_active = False
        # Una persona tiene prioridad. No se confia en una ruta si LiDAR ve objetos nuevos.
        target_speed, target_omega, safety_hit, safety_reason = safe_drive(target_speed, target_omega, environment, robot_node, person_detected or (person_clear_count < PERSON_CLEAR_DETECTIONS and step_counter > YOLO_EVERY_STEPS))
        if recovery_active and safety_reason != 'OK':
            print('RECUPERACION ABORTADA POR SEGURIDAD: {}'.format(safety_reason))
            recovery_active = False
            navigation_active = False
            route = []
            route_index = 0
            target_speed = 0.0
            target_omega = 0.0
        if safety_reason != 'OK' and step_counter % 60 == 0:
            print('SEGURIDAD: {} | persona {} | rayos frente {} atras {}'.format(safety_reason, person_detected, environment['front_count'], environment['back_count']))
        actual_omega = target_omega * TURN_SIGN
        left_velocity = (target_speed - actual_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
        right_velocity = (target_speed + actual_omega * DISTANCE_TO_CENTER) / WHEEL_RADIUS
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)

        if robot.getTime() - last_save_time >= SAVE_EVERY_SIM_SECONDS and mapping_active:
            memoria.save(MAP_FILE)
            last_save_time = robot.getTime()
            print('MEMORIA GUARDADA | libres {}, ocupadas {}, desconocidas {}'.format(*memoria.stats()))
finally:
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    memoria.save(MAP_FILE)
    cv2.destroyAllWindows()
