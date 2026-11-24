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

### Design principles (keep these intact)

- **Any future AI is an advisor, never a motor controller.** It may pick tasks or
  directions, but the LiDAR checks and the final safety layer always have the last word.
  AI calls must run off the main loop (background thread) and never call Webots APIs.
- **Fail safe**: no side with clearance, too many avoid attempts
  (`MAX_AVOID_ATTEMPTS`) or a stuck avoidance (`AVOID_MAX_SECONDS`) ends in STOP.
- Local reactive navigation only. There is no global planner or map.
- Known issue: the LiDAR sectors are cut by scan fractions and may be mislabeled
  (see `docs/mejoras-y-futuro.md`); fixing them by real ray angle is a pending step.

### Calibration knobs (top of the file)

`ROBOT_FORWARD_AXIS`, `TURN_SIGN`, `LIDAR_SWAP_SIDES` fix axis or orientation mismatches.
The `SAFE_DISTANCE_*` / `*_CLEARANCE*` values are conservative starting margins that
should be tuned to the robot's real footprint. LiDAR sector fractions in
`read_lidar_distances()` should be checked visually in Webots.

## Running

1. Install Webots R2025a and open `mir100/worlds/mir100.wbt`. EXTERNPROTOs are fetched
   from GitHub, so the first load needs internet.
2. Install the Python deps into the Python interpreter Webots uses (set in Webots
   preferences): `ultralytics`, `opencv-python`, `numpy`.
3. Optional: API keys for future AI features go in the repo-root `.env` (gitignored).
   `load_env_file()` in the controller loads it at startup without overriding
   variables already set in the environment. Nothing needs a key today.
4. Click inside the 3D view, then press `1`, `2` or `3`.

There are no automated tests. Verify behavior by running the simulation and reading the
Webots console.

## Notes / gotchas

- Don't commit API keys. Keep them in the environment or `.env`.
- `keyboard_matlab.m` defines `function keyboard_control` (its name doesn't match the
  file) and is stock sample code. Ignore it unless it's being revived.
- `.mir100.wbproj` / `.mir100.jpg` are Webots-generated UI state and thumbnail files.
