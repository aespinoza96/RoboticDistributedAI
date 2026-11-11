# RoboticDistributedAI

A Webots (R2025a) simulation of a **MiR100** autonomous mobile robot in an industrial
factory. The robot drives to delivery points on its own, using LiDAR for obstacle safety,
YOLO for person detection, and an external LLM ("JEV", via the TypeSafe SDK) as an
*advisor* that suggests which way to turn when the path is blocked.

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
│   │   └── yolo11n.pt                   YOLO11 nano weights (COCO), loaded by name
│   ├── keyboard_matlab/keyboard_matlab.m  Leftover stock Webots MATLAB keyboard teleop (unused)
│   └── Makefile                         Stock Webots Makefile (not needed for Python)
└── webots.yaml
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

1. **LiDAR**: front scan split into sectors `left / front_left / front / front_right /
   right`, back scan gives `back`. A sector with too few valid rays reads `0.0` and
   `ok=False`. *Unknown means blocked*: never treat missing data as free space.
2. **YOLO** (every `YOLO_EVERY_N_STEPS`): any COCO class 0 (person) above
   `YOLO_CONFIDENCE` puts the robot in `WAIT_PERSON` (full stop). It resumes after
   `PERSON_CLEAR_DETECTIONS` clean inferences. Shows an OpenCV window with detections.
3. **Keyboard**: `1/2/3` start a route to ENTREGA_1/2/3, `0`/space cancel, arrows give
   manual control (only when not navigating).
4. **State machine** (`mode`):
   - `GOAL`: rotate in place toward the goal, then drive forward while
     `forward_corridor_clear()`. If blocked, ask JEV.
   - `JEV_WAIT`: motors stopped; a background thread asks JEV (TypeSafe
     `system_one`, `Choice` LEFT/RIGHT/STOP). The main loop never blocks; it polls the
     queue and times out after `JEV_REQUEST_TIMEOUT`. Late replies are discarded via
     `jev_request_id`.
   - `AVOID_TURN`: turn in place in the approved direction until the front is clear.
   - `AVOID_FOLLOW`: follow the obstacle edge (P-control on side distance) with a small
     goal bias, then return to `GOAL` after `AVOID_MIN_TRAVEL` and when the robot is
     facing the goal again.
   - `WAIT_PERSON`: stop; saves and restores the previous mode.
5. **Final safety layer**: runs after all of the above and overrides it. No forward
   motion if the corridor isn't clear, no in-place turn without turn clearance, no
   reverse if `back` is too close, full stop on a person or bad LiDAR.
6. Differential drive: `(v ∓ ω·DISTANCE_TO_CENTER) / WHEEL_RADIUS` (left / right).

### Design principles (keep these intact)

- **JEV is an advisor, never a motor controller.** Its answer is accepted only if
  confidence ≥ `JEV_MIN_CONFIDENCE`, the choice is LEFT/RIGHT, and the LiDAR confirms
  clearance on that side. Otherwise: STOP and cancel navigation.
- **Fail safe**: any error, timeout, missing SDK/API key, too many avoid attempts
  (`MAX_AVOID_ATTEMPTS`) or a stuck avoidance (`AVOID_MAX_SECONDS`) ends in STOP.
- Local reactive navigation only. There is no global planner or map.
- Never call Webots APIs from the JEV worker thread.

### Calibration knobs (top of the file)

`ROBOT_FORWARD_AXIS`, `TURN_SIGN`, `LIDAR_SWAP_SIDES` fix axis or orientation mismatches.
The `SAFE_DISTANCE_*` / `*_CLEARANCE*` values are conservative starting margins that
should be tuned to the robot's real footprint. LiDAR sector fractions in
`read_lidar_distances()` should be checked visually in Webots.

## Running

1. Install Webots R2025a and open `mir100/worlds/mir100.wbt`. EXTERNPROTOs are fetched
   from GitHub, so the first load needs internet.
2. Install the Python deps into the Python interpreter Webots uses (set in Webots
   preferences): `ultralytics`, `opencv-python`, `numpy`, `typesafe-sdk` (optional).
3. For JEV obstacle avoidance, put `TYPESAFE_API_KEY=...` in the repo-root `.env`
   (gitignored). `load_env_file()` in the controller loads it at startup without
   overriding variables already set in the environment. Without the key the robot still drives and stops safely at obstacles, but
   it won't go around them.
4. Click inside the 3D view, then press `1`, `2` or `3`.

There are no automated tests. Verify behavior by running the simulation and reading the
Webots console.

## Notes / gotchas

- Don't commit API keys. JEV reads the key from `TYPESAFE_API_KEY` (env or `.env`).
- `keyboard_matlab.m` defines `function keyboard_control` (its name doesn't match the
  file) and is stock sample code. Ignore it unless it's being revived.
- `.mir100.wbproj` / `.mir100.jpg` are Webots-generated UI state and thumbnail files.

## Experiments (`experimentos/`)

A/B benchmark of the obstacle-avoidance decider, run headless (Webots `--batch --mode=fast --no-rendering`; works while a GUI Webots is open, on a different `--port`).

- `mir100/controllers/keyboard_control/experimento.py`: experiment mode, enabled only by `EXP_*` env vars (normal keyboard use is unchanged). Deciders: `jev` (original behaviour), `regla` (side with more clearance, tie-break toward the goal, same LiDAR acceptance filter), `sin_jev` (original controller with Jev unavailable). Each rep > 0 shifts the start pose (±0.3 m, ±0.6 rad, fixed seed per route+rep), so every decider faces the same scenarios. Each run appends one JSON line and calls `simulationQuit`.
- `JEV_PROVEEDOR=typesafe` (default, official SDK, `api.typesafe.ai`) or `jevai` (`https://www.jevai.org/api/v1/decisions`, personal keys from jevai.org). Both read `TYPESAFE_API_KEY` from `.env`.
- Run: `python3 experimentos/ejecutar.py --decisores regla sin_jev jev --rutas 1 2 3 --reps 10` (it checks Jev with one real call first and skips Jev runs if it fails). Analyse: `python3 experimentos/analizar.py` → CSVs, `benchmark.md`, `benchmark.png` in `experimentos/resultados/`.
- Results of 2026-11-10 (90 runs): all three deciders reach the goal only on route 2 (0.8 m, no obstacle). Jev answered RIGHT 20/20 (same as the rule) with confidence 0.19–0.35 < `JEV_MIN_CONFIDENCE` 0.60, so all were rejected and `jev` ≡ `sin_jev` in all 30 paired scenarios (+0.53 s mean latency). The rule's turns were accepted but `AVOID_FOLLOW` failed 20/20 (`EVASION SIN SALIDA`); the LiDAR sector fractions are mislabeled (see the PRD), which is the likely root cause.
