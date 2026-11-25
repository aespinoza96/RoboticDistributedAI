# RoboticDistributedAI

A Webots (R2025a) simulation of a **MiR100** autonomous mobile robot in an industrial
factory. The robot drives to delivery points on its own, using LiDAR for obstacle safety,
YOLO for person detection, and a local LiDAR rule that picks which way to turn when the
path is blocked. (An earlier external LLM advisor, "JEV", was removed: in the A/B
experiment it did no better than the rule.)

Code comments, console logs and identifiers are in **Spanish**. Keep that convention
when editing (e.g. `DESTINO ALCANZADO`, `PERSONA DETECTADA`).

## Layout

```
mir100/                                  Webots project root (open this in Webots)
├── worlds/mir100.wbt                    Factory world: MiR100, 3 delivery boxes, pedestrian, UR arms
├── protos/JetBotRaspberryPiCamera.proto Local camera PROTO mounted on the MiR100 body
├── controllers/
│   ├── keyboard_control/
│   │   ├── keyboard_control.py          MAIN controller (the one that matters)
│   │   ├── puente.py                    Command bridge: local TCP server, JSON lines
│   │   ├── lidar.py                     LiDAR sectors by real ray angle (pure numpy)
│   │   ├── profundidad.py               Depth camera → 3D obstacles above the LiDAR plane
│   │   ├── mapa.py                      Occupancy map (LiDAR layer + depth 'alto' layer)
│   │   ├── planificador.py              A* route + pure pursuit (Navegador), keep-out zones
│   │   ├── personas.py                  Locate YOLO people (bearing + depth distance)
│   │   └── yolo11n.pt                   YOLO11 nano weights (COCO), loaded by name
│   ├── keyboard_matlab/keyboard_matlab.m  Leftover stock Webots MATLAB keyboard teleop (unused)
│   └── Makefile                         Stock Webots Makefile (not needed for Python)
└── webots.yaml
agente/
├── robot_cliente.py                     Python client for the bridge
├── herramientas.py                      Robot tools for agents (wait for task end, failure hints)
├── servidor_mcp.py                      MCP server (stdio) exposing those tools
├── agente.py                            Autonomous agent: Claude + MCP tools, loop to success/failure
├── requirements.txt                     mcp>=2.3 (SDK 2.x: MCPServer, not FastMCP), anthropic[mcp]>=2.14
└── cli.py                               Manual bridge testing from a terminal, no AI
.mcp.json                                Registers the mir100 MCP server for Claude Code
tests/                                   Bridge, tools, MCP, agent, LiDAR and depth tests, no Webots needed
docs/mejoras-y-futuro.md                 Roadmap (Spanish)
```

## The world (`mir100.wbt`)

- `Mir100` with `supervisor TRUE` (required: the controller reads its own pose and the
  destination poses via the Supervisor API). It has no explicit `controller` field, so
  it uses the PROTO default `keyboard_control`, which resolves to the local controller.
- Devices: `front_lidar` / `back_lidar` (SickS300), `camera` (JetBot RPi camera),
  wheel motors `middle_left_wheel_joint` / `middle_right_wheel_joint`.
- Body carries a Robotino3 platform + `ConveyorPlatform` (controller
  `keyboard_conveyor_platform`, a stock Webots controller, not in this repo).
- Destinations: `DEF ENTREGA_1`, `ENTREGA_2`, `ENTREGA_3` (SolidBoxes). The controller
  looks them up with `getFromDef`, so do not rename these DEFs.
- A `Pedestrian` (to trigger the person stop) and UR3e/UR5e/UR10e arms running the stock
  `ure_can_grasper` controller.
- World uses **Z-up**.

## Main controller: `keyboard_control.py`

Runs one `robot.step()` loop. Each step:

1. **LiDAR** (`lidar.py`): both SickS300 scans become points in the robot frame by
   real ray angle (they sit on opposite corners, rotated and upside down: front ray
   *i* → −90° + *i*·step, back ray *i* → +90° + *i*·step; 0° = forward, + = left).
   Sectors `left / front_left / front / front_right / right / back` are angle ranges
   seen from the robot center, and each value is **clearance from the robot's
   footprint edge** (890×580 mm), not from the sensor. Never cut the range image by
   percentages. A sector with too few valid rays reads `0.0` and `ok=False`.
   *Unknown means blocked*: never treat missing data as free space.
   **Depth camera** (`profundidad.py`): the LiDAR plane (~0.19 m) misses tabletops,
   raised conveyors and anything above it. The `depth_camera` RangeFinder (x 0.464,
   z 0.18, forward, 118°, 0.18–1.95 m; Webots returns **planar** depth, verified: the
   floor reconstructs at z = 0.002 m) is projected to 3D points every
   `PROFUNDIDAD_CADA_N_PASOS`; points between 0.05 m and `ALTURA_ROBOT` (1.0 m) are
   obstacles. Each sector takes the **minimum** of LiDAR and depth clearance, so the
   whole safety/avoidance logic sees them. Depth only adds obstacles, never makes a
   sector unknown. Sectors where depth is clearly closer than the LiDAR are reported as
   `obstaculos_altos` (in `estado`/`escanear`) and as an `obstaculo_alto` event.
   `MIR_PROFUNDIDAD=0` turns it off for A/B runs; `volcar_profundidad(ruta=.npy)` dumps
   the raw image for calibration. Verified at the world's table: tabletop seen at
   0.67–0.69 m height where the LiDAR read 0.2–0.3 m farther.
2. **YOLO + people** (every `YOLO_EVERY_N_STEPS`; camera 640×480, 90°, `far 10` in the
   world, was 1 m). People come from a **dedicated detector** (`DETECTORES_PERSONAS`, first
   file found): `yoloe_personas.pt` (YOLOE-26s with the class "person" fixed, threshold
   0.15) else `yolo_personas.pt` (YOLO-World) else YOLO11n at 0.35. Built locally with
   `construir_detector_personas.py` (not in git; downloads YOLOE weights + MobileCLIP once).
   Measured on 39 Webots images of the pedestrian (front/profile/back, 1.3–3.8 m): YOLO11n
   4/39, YOLO-World 14/39, YOLO26s 17/39, YOLOE-26s 23/39. Boxes must be taller than wide
   (`PERSONA_PROPORCION_MIN`). YOLO11n still labels other objects in the window.
   Each person box is located by `personas.ubicar` (bearing from the box;
   distance from depth points inside the box, else LiDAR points, else box height marked
   `tamano_aproximado`). Detection alone never stops the robot: the rule is **distance**.
   - **People memory** (`remember_person` / `remembered_people`): detections are kept in
     world coordinates for `PERSONA_RECUERDO_S` (10 s) and the stop is evaluated **every
     step** against them. Up close the camera only sees legs and, when passing beside, the
     person leaves the frame: without memory the robot "forgot" them and drove past.
     Size-only estimates count only while visible.
   - **Signals to people** (`avisos.py`): the MiR100's light bands (PROTO field `color`,
     set live by the Supervisor) are cyan normally, **yellow** with a person within 2 m,
     **blinking yellow** while asking for way, **orange** if the person didn't move. A beep
     plays when someone enters 2 m; on a stop the robot says "Hola. Por favor, muévase del
     camino para que pueda continuar." (macOS `say`, voice `MIR_VOZ`, default Paulina),
     repeats every 20 s, and says "Gracias. Continúo." when it resumes. Sound plays on the
     host running Webots; `MIR_SONIDO=0` mutes it.
   - **Priority stop:** a person at `PERSONA_PARADA_M` (1.0 m) clearance or less, or
     `PERSONA_PARADA_FRENTE_M` (1.5 m) when ahead (±35°, `person_stop_distance`), stops the
     robot. The frontal margin exists because the LiDAR puts the person's legs in the map
     and the planner would otherwise swerve before reaching 1 m. Verified in Webots: stopped
     1.50 m in front of the pedestrian, located within ~0.15 m of its real position. The
     YOLO window shows each person's measured clearance and source (red = STOP), and the
     controller prints `VERSION DEL CONTROLADOR` at start. The stop and starts a `PERSONA_ESPERA_S` (60 s, sim time) wait in
     `WAIT_PERSON`; event `persona_cerca` (distance, angle, world position). The task
     **resumes as soon as the person moves away** (no trace from the camera or LiDAR
     echoes within `PERSONA_RASTRO_M` of where they stood for `PERSONA_LIBRE_S`, 2 s;
     event `persona_se_aparto`). If they are still there after `PERSONA_ESPERA_S` (60 s) it
     ends `detenida` / `persona_sigue_cerca`. Verified in Webots with a walking pedestrian:
     stopped at 1.5 m, then resumed ~7 s later when they had walked away. New orders during the wait also wait. Fires once per
     encounter; re-arms beyond `PERSONA_REARME_M` (1.3 m) or out of view.
   - Under `PERSONA_LENTO_M` (2 m) speed is capped at `PERSONA_VELOCIDAD`; forward motion
     is blocked for a person ahead (±`PERSONA_ANGULO_FRENTE`) within the stop distance.
   - **The route does not change for people by default** (`al_ver_persona="continuar"`):
     keep-out zones used to close corridors (worse with size-estimated distances) so the
     robot never approached. With `al_ver_persona="rodear"` people become 1.0 m zones
     (`PERSONA_ZONA_RADIO`, remembered `PERSONA_MEMORIA_S`) the planner avoids.
   Shows an OpenCV window with detections.
3. **Keyboard**: `1/2/3` start a route to ENTREGA_1/2/3, `0`/space cancel, arrows give
   manual control (only when not navigating).
4. **State machine** (`mode`):
   - `GOAL`: rotate in place toward the goal, then drive forward while
     `forward_corridor_clear()`. If blocked, `choose_avoid_direction(scan)` picks the
     side that has turn clearance (side and diagonal) and the most free space; if
     neither side qualifies, STOP and cancel.
   - `AVOID_TURN`: turn in place in the chosen direction until the front is clear.
   - `AVOID_FOLLOW`: follow the obstacle edge (P-control on side distance) with a small
     goal bias, then return to `GOAL` after `AVOID_MIN_TRAVEL` and when the robot is
     facing the goal again.
   - `WAIT_PERSON`: stop; saves and restores the previous mode.
5. **Final safety layer**: runs after all of the above and overrides it. No forward
   motion if the corridor isn't clear, no in-place turn without turn clearance, no
   reverse if `back` is too close, full stop on a person or bad LiDAR.
6. Differential drive: `(v ∓ ω·DISTANCE_TO_CENTER) / WHEEL_RADIUS` (left / right).

### Planned navigation (`mapa.py`, `planificador.py`)

`ir_a` / `ir_a_punto` (and keys 1/2/3) run in `MODE_PLAN` when `PLANIFICADOR_ACTIVO`
(`MIR_PLANIFICADOR=0` falls back to the reactive GOAL/AVOID modes). The occupancy map
(0.1 m cells, −10..10 × −7..7) is updated every `MAPA_CADA_N_PASOS` from both LiDARs
(free + hit, log-odds) and from depth obstacle points in a separate `alto` layer that
LiDAR rays can't erase (they pass under tables). The map persists across tasks.
`Navegador` plans A* with obstacles inflated 0.55 m (soft cost to 0.95 m), unknown cells
treated as free, replans every 2 s or when the path becomes occupied, and follows the
path with pure pursuit. In plan mode the final safety layer uses collision margins
(`PLAN_*_MIN`) instead of the conservative reactive ones, keeps turning allowed when
forward is blocked, and forces a replan. New outcomes: `sin_ruta` (no free path on the
map) and `meta_inalcanzable` (goal inside an obstacle; reached the closest free cell).
Verified once in Webots: ENTREGA_1 reached from the start (around the 8 m conveyor).
LiDAR hits beyond `MAX_RANGO_IMPACTO` (4 m) only clear free space: sparse far hits,
inflated, used to join into walls that closed real corridors (false `sin_ruta`). The
planner retries with `INFLADO_MINIMO` (0.38 m) before reporting `sin_ruta`.
`volcar_mapa` dumps the layers for diagnosis.

### Tasks and the command bridge

Every order (keyboard `1/2/3` or bridge) is a **task** that always ends explicitly with
`estado` = `exito` / `fallo` / `cancelada` and a machine-readable `motivo`, plus a
`detalle` text and a `contexto` snapshot (pose, mode, LiDAR sectors, person). A new
order cancels the current task (`motivo: reemplazada`).

`puente.py` listens on `127.0.0.1:5555` (`MIR_PUENTE_PUERTO` overrides). One JSON object
per line. `bridge.poll()` runs once per `robot.step()` in the main loop; nothing blocks
and no other thread touches Webots.

- Request `{"id": 1, "cmd": "ir_a", "args": {"destino": "ENTREGA_3"}}` →
  `{"tipo": "respuesta", "id": 1, "ok": true, "resultado": {...}}` (or `ok: false`, `error`).
- Commands: `estado`, `destinos`, `escanear`, `ruta(destino | x, y)` (A* route length on the
  current map, no motion), `ir_a(destino)`, `ir_a_punto(x, y)`
  (waypoint in world coordinates, same navigation as `ir_a`, |x|,|y| ≤ 50),
  `girar(grados)` (+ = left, max 180), `avanzar(metros)` (negative = reverse, max 1.0),
  `parar`, `volcar_profundidad(ruta)`, `volcar_mapa(ruta.npz)`, `volcar_camara(ruta.png, cruda)`
  (camera frame + YOLO11n detections at 0.10, for diagnosis). Movement commands return a `tarea_id`
  immediately; the result arrives as an event.
- **Per-order options** on `ir_a`, `ir_a_punto`, `girar`, `avanzar` (`parse_task_options`):
  `velocidad` (0.05–0.40 m/s, plan speed), `al_ver_persona` = `continuar` (default) | `rodear` |
  `detenerse` (task ends with `estado: "detenida"`, `motivo: "persona_a_la_vista"` as
  soon as a person is within `distancia_persona`) | `esperar` (forward stops while a
  person is ahead within `distancia_persona`, also in plan mode, then resumes), and
  `distancia_persona` (0.3–10 m, default 3). Non-default options are recorded in the
  task's `args`. People are located with depth, then **LiDAR** points inside the box
  (legs cut the scan plane), then box height (`fuente: "tamano_aproximado"`, capped at
  the camera's 10 m); `contexto.personas` includes world `posicion`.
- Events `{"tipo": "evento", "evento": ..., "t": sim_seconds, "datos": {...}}` go to all
  clients: `tarea_iniciada`, `tarea_terminada`, `evasion_iniciada`, `evasion_terminada`,
  `persona_detectada` (first sighting, with clearance/angle), `persona_en_el_camino`,
  `persona_despejada`, `persona_bloqueando` (warning only, after
  `PERSONA_BLOQUEANDO_SEGUNDOS`), `lidar_sin_datos`.
- Failure reasons (`motivo`): `sin_espacio_lateral`, `limite_evasiones`,
  `evasion_sin_salida` (`AVOID_MAX_SECONDS`, 25 s), `bloqueado_por_seguridad` (the
  safety layer zeroed the navigation command for `BLOQUEO_NAVEGACION_SEGUNDOS`, 5 s;
  `detalle` names the rule and mode), `sin_progreso` (watchdog: navigating without
  moving `SIN_PROGRESO_DISTANCIA` in `SIN_PROGRESO_SEGUNDOS`, 20 s), `lidar_sin_datos`,
  `maniobra_bloqueada` (the safety layer blocked a maneuver; `detalle` names the rule),
  `tiempo_agotado`. Success: `llego`, `maniobra_completa`.
- Maneuvers go through the same final safety layer as everything else. Waiting for a
  person pauses the maneuver timer and the progress watchdog.

### MCP server (`agente/servidor_mcp.py`)

Tools: `estado_robot`, `listar_destinos`, `escanear_entorno`, `distancia_por_ruta`, `ir_a`, `ir_a_punto`,
`girar`, `avanzar`, `parar`, `esperar_tarea`. Movement tools take the per-order options
above, plus `interrumpir_con` (event names, e.g. `persona_detectada`, `obstaculo_alto`):
the wait returns early with `estado: "en_curso"`, `interrumpida_por` and the event while
the task keeps running, so the agent can decide mid-task. They wait (default) until the task ends
and return `estado`, `motivo`, `detalle`, `contexto`, the event timeline and a
`sugerencia` from `herramientas.SUGERENCIAS`. Each tool call opens its own bridge
connection and runs in a worker thread, so `parar` works while `ir_a` is waiting.
Errors (no Webots, bad args) come back as `{"error": ...}` rather than exceptions.
Logic lives in `herramientas.py` so the part-3 agent can reuse it.

### Autonomous agent (`agente/agente.py`)

`.venv/bin/python agente/agente.py "Vaya al punto de entrega 3"`. Launches
`servidor_mcp.py` over stdio, converts its tools with `anthropic.lib.tools.mcp.async_mcp_tool`
and runs the Anthropic SDK tool runner (`claude-opus-5-5`, adaptive thinking, effort
`high`, server-side `fallbacks: "default"`, auto prompt caching). The model plans,
acts, recovers and must finish by calling the local `reportar_resultado` tool.

Hard limits enforced in code, not by the prompt:
- **Progress, not count:** after each movement result the program asks the robot for the
  **A\* route length** to the current `ir_a` goal (`distancia_por_ruta`; straight line
  if unavailable) and the known map area. Shortening the route by `MEJORA_MINIMA_M`
  (0.3 m) **or** discovering `DESCUBRIMIENTO_MINIMO_M2` (2 m²) of new map resets the
  counter (discovery also resets the baseline, since new walls lengthen routes). A detour
  that moves away in a straight line but shortens the route is progress. Each failed
  navigation (`ir_a`, `ir_a_punto`, `esperar_tarea`) with neither is one strike;
  `sin_ruta` never counts (it doesn't move; the prompt says to explore). At
  `--max-sin-progreso` (3) strikes, movement tools are refused. Every movement result
  gets a "Progreso: ..." line appended so the model sees its budget.
- Cost guards: `--max-movimientos` (30), `--max-pasos` (60 model turns), `--minutos` (15).
The prompt tells the model to plan detours with `ir_a_punto` around `obstaculos_altos`
and the free side, and to use one maneuver with the total value instead of many small ones. If the model never
reports (limit, API error, refusal), the program writes a fallback report from the last
robot result and its `sugerencia`. Any non-success ends with `parar`. Checks the robot
connection before the first API call. Each run is logged to `agente/registros/` (gitignored).
Uses its own venv (`.venv`, gitignored) because the system Python has `anthropic` 0.x.
`tests/test_agente.py` drives it with a scripted fake tool runner (no API calls); it is
skipped under the system Python.

### Design principles (keep these intact)

- **Any future AI is an advisor, never a motor controller.** It may pick tasks or
  directions, but the LiDAR checks and the final safety layer always have the last word.
  AI calls must run off the main loop (background thread) and never call Webots APIs.
- **Fail safe**: no side with clearance, too many avoid attempts
  (`MAX_AVOID_ATTEMPTS`) or a stuck avoidance (`AVOID_MAX_SECONDS`) ends in STOP.
- Local reactive navigation only. There is no global planner or map.
- Known issue: reactive avoidance still fails to reach ENTREGA_1 and ENTREGA_3
  (`evasion_sin_salida` / `sin_progreso` while edge-following). Next options: tune the
  edge-following and side thresholds, or bring the map + A\* from the previous project.

### Calibration knobs (top of the file)

`ROBOT_FORWARD_AXIS`, `TURN_SIGN`, `LIDAR_SWAP_SIDES` fix axis or orientation mismatches.
The `SAFE_DISTANCE_*` / `*_CLEARANCE*` values are conservative starting margins that
are now measured as clearance from the robot body (before, the "right" value included
the robot's width), so they may need re-tuning. LiDAR mounting angles, positions,
footprint and sector ranges live at the top of `lidar.py`; they were verified in
Webots against the arena walls (right 1.22 m vs 1.26 expected, back 1.08 vs 1.1).

## Running

1. Install Webots R2025a and open `mir100/worlds/mir100.wbt`. EXTERNPROTOs are fetched
   from GitHub, so the first load needs internet.
2. Install the Python deps into the Python interpreter Webots uses (set in Webots
   preferences): `ultralytics`, `opencv-python`, `numpy`.
3. Optional: API keys for future AI features go in the repo-root `.env` (gitignored).
   `load_env_file()` in the controller loads it at startup without overriding
   variables already set in the environment. Nothing needs a key today.
4. Click inside the 3D view, then press `1`, `2` or `3`, or drive it through the bridge:
   `python3 agente/cli.py ir_a destino=ENTREGA_3 --esperar` (see `agente/cli.py`).
5. MCP: `python3 -m pip install -r agente/requirements.txt`. Claude Code picks up
   `.mcp.json` from the repo root (approve the `mir100` server once); then ask it in
   plain language, e.g. "lleva el robot a la entrega 3".
6. Autonomous agent: `python3 -m venv .venv && .venv/bin/python -m pip install -r
   agente/requirements.txt`, put `ANTHROPIC_API_KEY=...` in `.env`, then
   `.venv/bin/python agente/agente.py "Vaya al punto de entrega 3"`. Each run bills API usage.

Tests (no Webots, no API): `.venv/bin/python -m unittest discover -s tests -v` (all)
or `python3 -m unittest discover -s tests -v` (agent tests skipped). Controller behavior is
verified by running the simulation and reading the Webots console. Headless run for
testing: `MIR_PUENTE_PUERTO=5560 /Applications/Webots.app/Contents/MacOS/webots --batch
--mode=fast --minimize --no-rendering --stdout --stderr --port=1240 mir100/worlds/mir100.wbt`
(a separate port pair avoids clashing with an open Webots).

## Notes / gotchas

- Don't commit API keys. Keep them in the environment or `.env`.
- `keyboard_matlab.m` defines `function keyboard_control` (its name doesn't match the
  file) and is stock sample code. Ignore it unless it's being revived.
- `.mir100.wbproj` / `.mir100.jpg` are Webots-generated UI state and thumbnail files.
