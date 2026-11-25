# Webots R2025a: mover EXCLUSIVAMENTE al Pedestrian.
# No modifica cajas, maquinaria, entregas ni el controlador del MiR100.
from controller import Supervisor
import math

TIME_STEP = 32
SPEED = 0.30
MIN_DISTANCE_TO_MIR = 1.25
PERSON_DEF = 'PERSONA_MOVIL'
ROBOT_DEF = 'MIR100'

# Coordenadas del suelo X/Y del escenario. Se conserva la altura Z.
# Puntos de ejemplo en el area inicial; revisar que no atraviesen maquinaria.
ROUTE = [(-2.51, -5.18), (-3.30, -4.60), (-3.70, -3.80), (-3.30, -4.60), (-2.51, -5.18)]

supervisor = Supervisor()
person = supervisor.getFromDef(PERSON_DEF)
robot = supervisor.getFromDef(ROBOT_DEF)

if person is None:
    raise RuntimeError('No se encontro DEF PERSONA_MOVIL en el mundo.')

translation = person.getField('translation')
if translation is None:
    raise RuntimeError('El Pedestrian no expone el campo translation.')

position = translation.getSFVec3f()
z = position[2]
next_point = 1
print('PEDESTRIAN MOVIL ACTIVO | El resto de objetos permanece inmovil')

while supervisor.step(TIME_STEP) != -1:
    if len(ROUTE) < 2:
        continue
    x, y, _ = translation.getSFVec3f()
    tx, ty = ROUTE[next_point]
    dx, dy = tx-x, ty-y
    distance = math.hypot(dx, dy)
    if distance < 0.035:
        next_point = (next_point+1) % len(ROUTE)
        continue
    step = min(SPEED*TIME_STEP/1000.0, distance)
    nx = x+dx/distance*step
    ny = y+dy/distance*step
    if robot is not None:
        robot_pos = robot.getPosition()
        if math.hypot(nx-robot_pos[0], ny-robot_pos[1]) < MIN_DISTANCE_TO_MIR:
            continue
    translation.setSFVec3f([nx, ny, z])
