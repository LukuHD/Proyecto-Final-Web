from collections import deque
import numpy as np


class MapMetrics:
    def __init__(self, dungeon_map):
        self.dungeon_map = np.array(dungeon_map)
        self.height, self.width = self.dungeon_map.shape

    def _floor_cells(self):
        positions = np.argwhere(self.dungeon_map == 1)
        return [tuple(pos) for pos in positions]

    def _neighbors(self, y, x):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < self.height and 0 <= nx < self.width:
                yield ny, nx

    def calculate_connectivity(self):
        floor_cells = self._floor_cells()
        if not floor_cells:
            return 0.0
        visited = set()
        frontier = deque([floor_cells[0]])
        visited.add(floor_cells[0])
        while frontier:
            cy, cx = frontier.popleft()
            for ny, nx in self._neighbors(cy, cx):
                if self.dungeon_map[ny, nx] == 1 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    frontier.append((ny, nx))
        total_floor = len(floor_cells)
        reachable = len(visited)
        return reachable / total_floor if total_floor else 0.0

    def calculate_density(self):
        total_cells = self.height * self.width
        floor_cells = np.count_nonzero(self.dungeon_map == 1)
        return floor_cells / total_cells if total_cells else 0.0

    def calculate_room_distribution(self):
        row_sums = self.dungeon_map.sum(axis=1)
        if not np.any(row_sums):
            return 0.0
        normalized = row_sums / row_sums.max()
        spread = 1.0 - np.std(normalized)
        return float(np.clip(spread, 0.0, 1.0))

    def evaluate_corridor_quality(self):
        floor_cells = self._floor_cells()
        if not floor_cells:
            return 0.0
        straight_sections = 0
        for y, x in floor_cells:
            neighbors = [(ny, nx) for ny, nx in self._neighbors(y, x) if self.dungeon_map[ny, nx] == 1]
            if len(neighbors) == 2:
                dy = neighbors[0][0] - y
                dx = neighbors[0][1] - x
                dy2 = neighbors[1][0] - y
                dx2 = neighbors[1][1] - x
                if dy == -dy2 and dx == -dx2:
                    straight_sections += 1
        return straight_sections / len(floor_cells)

    def detect_dead_ends(self):
        floor_cells = self._floor_cells()
        if not floor_cells:
            return 1.0
        dead_ends = 0
        for y, x in floor_cells:
            neighbors = sum(1 for ny, nx in self._neighbors(y, x) if self.dungeon_map[ny, nx] == 1)
            if neighbors == 1:
                dead_ends += 1
        ratio = dead_ends / len(floor_cells)
        return 1.0 - ratio

    def as_dict(self):
        return {
            'connectivity': self.calculate_connectivity(),
            'density': self.calculate_density(),
            'room_distribution': self.calculate_room_distribution(),
            'corridor_quality': self.evaluate_corridor_quality(),
            'dead_ends': self.detect_dead_ends(),
        }

# Example usage:
# dungeon = ...  # Some structure representing the dungeon map
# metrics = MapMetrics(dungeon)
# metrics.calculate_connectivity()