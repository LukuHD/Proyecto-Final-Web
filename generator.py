import random
from collections import deque

from la import MapGenerator


def connect_rooms(room1, room2, grid):
    """Talla un pasillo en forma de L entre los centros de dos habitaciones."""
    center1_x = room1['x'] + room1['width'] // 2
    center1_y = room1['y'] + room1['height'] // 2
    center2_x = room2['x'] + room2['width'] // 2
    center2_y = room2['y'] + room2['height'] // 2

    if random.choice([True, False]):
        # Horizontal, luego Vertical
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1):
            grid[center1_y, x] = 1 # 1 representa el suelo
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1):
            grid[y, center2_x] = 1
    else:
        # Vertical, luego Horizontal
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1):
            grid[y, center1_x] = 1
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1):
            grid[center2_y, x] = 1

def _flood_fill_components(grid):
    height, width = grid.shape
    visited = set()
    components = []
    for y in range(height):
        for x in range(width):
            if grid[y, x] != 1 or (y, x) in visited:
                continue
            frontier = deque([(y, x)])
            current_component = []
            visited.add((y, x))
            while frontier:
                cy, cx = frontier.popleft()
                current_component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] == 1 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        frontier.append((ny, nx))
            components.append(current_component)
    return components


def _connect_components(grid):
    components = _flood_fill_components(grid)
    if len(components) <= 1:
        return
    primary = components[0]
    primary_set = set(primary)
    for extra_component in components[1:]:
        best_pair = None
        best_distance = None
        for cell_a in primary:
            for cell_b in extra_component:
                distance = abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (cell_a, cell_b)
        if best_pair is None:
            continue
        (ay, ax), (by, bx) = best_pair
        pseudo_room_a = {'x': ax, 'y': ay, 'width': 1, 'height': 1}
        pseudo_room_b = {'x': bx, 'y': by, 'width': 1, 'height': 1}
        connect_rooms(pseudo_room_a, pseudo_room_b, grid)
        primary_set.update(extra_component)
        primary = list(primary_set)
    # Validar nuevamente por si quedan componentes separados
    remaining = _flood_fill_components(grid)
    if len(remaining) > 1:
        _connect_components(grid)


def create_dungeon_layout(config: dict):
    """Genera mapas usando las estrategias avanzadas definidas en ``la.py``."""
    environment_type = config.get('environment_type', 'dungeon')
    generator = MapGenerator(environment_type=environment_type)

    key_map = {
        'width': 'map_width',
        'height': 'map_height',
        'min_leaf_size': 'min_leaf_size',
        'room_min_size': 'room_min_size',
        'padding': 'room_padding',
    }

    overrides = {target: config[source] for source, target in key_map.items() if source in config}
    extra_overrides = config.get('overrides')
    if isinstance(extra_overrides, dict):
        overrides.update(extra_overrides)

    if overrides:
        generator.config.update(overrides)

    dungeon_grid = generator.generate()
    _connect_components(dungeon_grid)
    return dungeon_grid