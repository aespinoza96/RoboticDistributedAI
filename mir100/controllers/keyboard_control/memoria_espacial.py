# Memoria espacial: ocupacion probabilistica y A*, Webots Z vertical / XY suelo.
# Independiente de Webots: permite probarla tambien con Python normal.
import os
import json
import math
import heapq

class MemoriaEspacial:
    def __init__(self, xmin=-9.0, ymin=-9.0, xmax=9.0, ymax=9.0, resolution=0.20, robot_radius=0.50):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.width = int(math.ceil((self.xmax - self.xmin) / self.resolution))
        self.height = int(math.ceil((self.ymax - self.ymin) / self.resolution))
        self.log_odds = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        self.revision = 0
        self.temporary_obstacles = {}

    def inside(self, c, r):
        return 0 <= c < self.width and 0 <= r < self.height

    def world_to_cell(self, x, y):
        c = int(math.floor((x - self.xmin) / self.resolution))
        r = int(math.floor((y - self.ymin) / self.resolution))
        return (c, r) if self.inside(c, r) else None

    def cell_to_world(self, c, r):
        return (self.xmin + (c + 0.5) * self.resolution, self.ymin + (r + 0.5) * self.resolution)

    def _add(self, c, r, delta):
        if self.inside(c, r):
            self.log_odds[r][c] = max(-5.0, min(5.0, self.log_odds[r][c] + delta))

    def observe_ray(self, origin_xy, angle, measured_distance, max_range, hit):
        # Trazo de rayo: solo se considera libre la parte observada; lo desconocido permanece desconocido.
        if not math.isfinite(measured_distance) or measured_distance <= 0.0:
            return
        ray_length = min(float(measured_distance), float(max_range))
        end_free = ray_length - (0.15 if hit else 0.0)
        if end_free < 0.0:
            return
        steps = max(1, int(end_free / (self.resolution * 0.65)))
        seen = set()
        for i in range(steps + 1):
            d = end_free * i / steps
            c = self.world_to_cell(origin_xy[0] + d * math.cos(angle), origin_xy[1] + d * math.sin(angle))
            if c is not None and c not in seen:
                self._add(c[0], c[1], -0.28)
                seen.add(c)
        if hit and measured_distance < max_range - 0.05:
            c = self.world_to_cell(origin_xy[0] + ray_length * math.cos(angle), origin_xy[1] + ray_length * math.sin(angle))
            if c is not None:
                self._add(c[0], c[1], 1.8)
        self.revision += 1

    def is_free(self, c, r):
        return self.inside(c, r) and self.log_odds[r][c] <= -0.8

    def mark_temporary_obstacle(self, x, y, radius=0.30, expires_at=0.0):
        """Bloqueo local provisional; no altera el mapa persistente."""
        center = self.world_to_cell(x, y)
        if center is None:
            return False
        cr = int(math.ceil(radius / self.resolution))
        changed = False
        for dr in range(-cr, cr + 1):
            for dc in range(-cr, cr + 1):
                if math.hypot(dc, dr) * self.resolution > radius:
                    continue
                cell = (center[0] + dc, center[1] + dr)
                if self.inside(*cell):
                    old = self.temporary_obstacles.get(cell, -1.0)
                    self.temporary_obstacles[cell] = max(old, expires_at)
                    changed = changed or old < 0
        if changed:
            self.revision += 1
        return changed

    def expire_temporary_obstacles(self, now):
        old = len(self.temporary_obstacles)
        self.temporary_obstacles = {cell: until for cell, until in self.temporary_obstacles.items() if until > now}
        if len(self.temporary_obstacles) != old:
            self.revision += 1

    def blocked_grid(self):
        # Desconocido = bloqueado. Se inflan celdas ocupadas y se respeta el borde del mapa.
        occupied = set()
        free = set()
        for r in range(self.height):
            for c in range(self.width):
                if self.log_odds[r][c] >= 0.8:
                    occupied.add((c, r))
                elif self.is_free(c, r):
                    free.add((c, r))
        radius_cells = int(math.ceil(self.robot_radius / self.resolution))
        inflation = set()
        occupied.update(self.temporary_obstacles.keys())
        for c, r in occupied:
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dc * dc + dr * dr <= radius_cells * radius_cells:
                        inflation.add((c + dc, r + dr))
        return free - inflation

    def cell_status(self, cell):
        if cell is None:
            return 'FUERA_DEL_MAPA'
        if cell in self.temporary_obstacles:
            return 'BLOQUEO_TEMPORAL'
        c, r = cell
        value = self.log_odds[r][c]
        if value >= 0.8:
            return 'OCUPADA'
        if value <= -0.8:
            return 'LIBRE'
        return 'DESCONOCIDA'

    def segment_is_clear(self, a, b, allow_start_unknown=False):
        free = self.blocked_grid()
        if not free:
            return False
        if allow_start_unknown:
            first = self.world_to_cell(a[0], a[1])
            if first is not None:
                free.add(first)
        return self.line_is_free(a, b, free)

    def _nearest_free(self, cell, free, radius=2):
        if cell in free:
            return cell
        if cell is None:
            return None
        candidates = [(math.hypot(c - cell[0], r - cell[1]), (c, r)) for c, r in free if abs(c - cell[0]) <= radius and abs(r - cell[1]) <= radius]
        return min(candidates)[1] if candidates else None

    def find_path(self, start_xy, goal_xy):
        free = self.blocked_grid()
        start = self._nearest_free(self.world_to_cell(start_xy[0], start_xy[1]), free)
        goal = self._nearest_free(self.world_to_cell(goal_xy[0], goal_xy[1]), free)
        if start is None or goal is None:
            return []
        heap = [(0.0, start)]
        g_score = {start: 0.0}
        parents = {}
        closed = set()
        while heap:
            _, current = heapq.heappop(heap)
            if current in closed:
                continue
            closed.add(current)
            if current == goal:
                path = [current]
                while path[-1] in parents:
                    path.append(parents[path[-1]])
                path.reverse()
                coords = [self.cell_to_world(c, r) for c, r in path]
                return self.simplify_path(coords, free)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if nxt not in free:
                    continue
                if dx and dy and ((current[0] + dx, current[1]) not in free or (current[0], current[1] + dy) not in free):
                    continue
                score = g_score[current] + math.hypot(dx, dy)
                if score >= g_score.get(nxt, float('inf')):
                    continue
                parents[nxt] = current
                g_score[nxt] = score
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(heap, (score + heuristic, nxt))
        return []

    def line_is_free(self, a, b, free):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(math.ceil(length / (self.resolution * 0.4))))
        for i in range(steps + 1):
            t = i / float(steps)
            cell = self.world_to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if cell not in free:
                return False
        return True

    def simplify_path(self, coords, free):
        if len(coords) <= 2:
            return coords
        result = [coords[0]]
        i = 0
        while i < len(coords) - 1:
            j = len(coords) - 1
            while j > i + 1 and not self.line_is_free(coords[i], coords[j], free):
                j -= 1
            result.append(coords[j])
            i = j
        return result

    def stats(self):
        free = sum(1 for row in self.log_odds for value in row if value <= -0.8)
        occupied = sum(1 for row in self.log_odds for value in row if value >= 0.8)
        return free, occupied, self.width * self.height - free - occupied

    def save(self, path):
        data = {'format': 1, 'bounds': [self.xmin, self.ymin, self.xmax, self.ymax], 'resolution': self.resolution, 'robot_radius': self.robot_radius, 'log_odds': self.log_odds}
        tmp = path + '.tmp'
        with open(tmp, 'w') as handle:
            json.dump(data, handle)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as handle:
            data = json.load(handle)
        item = cls(data['bounds'][0], data['bounds'][1], data['bounds'][2], data['bounds'][3], data['resolution'], data['robot_radius'])
        if len(data['log_odds']) != item.height or any(len(row) != item.width for row in data['log_odds']):
            raise ValueError('El archivo del mapa tiene dimensiones incorrectas')
        item.log_odds = data['log_odds']
        return item
