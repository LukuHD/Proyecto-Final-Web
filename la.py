import random
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
import heapq

# --- 1. CONFIGURACIONES DE ENTORNOS ---
ENVIRONMENT_PRESETS = {
    'dungeon': {
        'map_width': 50, 'map_height': 30, 'min_leaf_size': 6, 'room_min_size': 4,
        'room_shape_biases': ['normal', 'wide', 'tall'],
        'generator_algorithm': 'bsp'
    },
    'forest': {
        'map_width': 71, 'map_height': 51,
        'generator_algorithm': 'hybrid_forest',
        'min_leaf_size': 12,
        'forest_density': 0.15,
        'path_width': 2,
        'extra_path_connections_prob': 0.25,
        'path_wiggle_room': 0.3,
        'num_entrances': 5,
        'entrance_width': 3,
        'clearing_carvers': [
            {'type': 'horizontal_ellipse', 'weight': 1},
            {'type': 'vertical_ellipse', 'weight': 3},
            {'type': 'circle', 'weight': 2},
            {'type': 'random_walk', 'weight': 4},
            {'type': 'rectangle', 'weight': 2},
            {'type': 'triangle', 'weight': 1},
            {'type': 'medium_horizontal_cavern', 'weight': 5}
        ],
        'ellipse_params': {
            'num_shapes': (6, 10), 'min_radius': 2, 'max_radius': 5,
            'stretch_factor': 1.8,
            'medium_stretch_factor': 2.5
        },
        'random_walk_carver': { 'steps': 150, 'brush_radius': 2 },
        'rectangle_carver': { 'num_shapes': (2, 5), 'min_width': 3, 'max_width': 8, 'min_height': 3, 'max_height': 8 },
        'triangle_carver': { 'num_shapes': (1, 3) },
        'cellular_automata_passes': 3,
        'cellular_automata_birth_limit': 4, 'cellular_automata_death_limit': 3
    },
    # --- NUEVO TIPO DE MAPA ENFOCADO EN CAMINOS Y OBSTÁCULOS ---
    'path_focused': {
        'map_width': 80, 'map_height': 25,
        'generator_algorithm': 'path_focused', # Usa la nueva estrategia
        'path_width': 4, # Ancho del camino principal
        'path_wiggle_room': 0.8, # Factor de sinuosidad del camino
        'num_bifurcations': 12,
        'bifurcation_length': (8, 20),
        'bifurcation_width': 2,
        # --- SECCIÓN DE CONFIGURACIÓN DE OBSTÁCULOS ---
        'obstacles': [
            #                     Tipo de obstáculo y sus parámetros
            {'type': 'rock_cluster', 'count': 20, 'min_radius': 1, 'max_radius': 1}, # Piedras pequeñas
            {'type': 'rock_cluster', 'count': 15, 'min_radius': 2, 'max_radius': 2}, # Piedras normales
            {'type': 'rock_cluster', 'count': 8, 'min_radius': 2, 'max_radius': 3},  # Piedras grandes
            {'type': 'rock_cluster', 'count': 10, 'min_radius': 2, 'max_radius': 3, 'stretch': 2.5}, # Piedras alargadas
            {'type': 'log', 'count': 15, 'min_length': 4, 'max_length': 8, 'thickness': 1} # Troncos
        ]
    }
}

# --- 2. INTERFACES ABSTRACTAS (CLASES BASE) ---

class MapGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, config): pass

class ClearingCarver(ABC):
    @abstractmethod
    def carve(self, leaf, grid, config): pass

    def _draw_ellipse(self, cx, cy, rx, ry, grid, config, value):
        if rx <= 0: rx = 1
        if ry <= 0: ry = 1
        for x_offset in range(-rx, rx + 1):
            for y_offset in range(-ry, ry + 1):
                x, y = cx + x_offset, cy + y_offset
                if 0 <= x < config['map_width'] and 0 <= y < config['map_height']:
                    if (x_offset**2 / rx**2) + (y_offset**2 / ry**2) <= 1:
                        grid[y, x] = value

class ObstaclePlacer(ABC):
    """Clase base para generadores de obstáculos."""
    @abstractmethod
    def place(self, grid, path_coords, config):
        """
        Coloca obstáculos en el mapa.
        :param grid: El grid del mapa.
        :param path_coords: Una lista de coordenadas (y, x) que forman el camino.
        :param config: La configuración específica para este obstáculo.
        """
        pass

# --- 3. CLASES CONCRETAS DE "TALLADORES" (PARA BOSQUE) ---

class BaseEllipseCarver(ClearingCarver):
    """Clase base con lógica común para todos los talladores de elipses."""
    def carve(self, leaf, grid, config):
        params = config['ellipse_params']
        num_shapes = random.randint(*params['num_shapes'])
        
        for _ in range(num_shapes):
            rx, ry = self._calculate_radii(leaf, params)
            if rx is None or ry is None: continue

            start_cx, end_cx = leaf.x + rx, leaf.x + leaf.width - rx - 1
            if start_cx > end_cx: continue
            start_cy, end_cy = leaf.y + ry, leaf.y + leaf.height - ry - 1
            if start_cy > end_cy: continue

            cx, cy = random.randint(start_cx, end_cx), random.randint(start_cy, end_cy)
            self._draw_ellipse(cx, cy, rx, ry, grid, config, 1)

    @abstractmethod
    def _calculate_radii(self, leaf, params):
        """Método que las clases hijas deben implementar para definir la forma."""
        pass

class LargeHorizontalCavernCarver(ClearingCarver):
    def carve(self, grid, config):
        map_width = config['map_width']
        map_height = config['map_height']
        cx = random.randint(int(map_width * 0.45), int(map_width * 0.55))
        cy = random.randint(int(map_height * 0.45), int(map_height * 0.55))
        rx = random.randint(int(map_width * 0.30), int(map_width * 0.40))
        ry = random.randint(int(map_height * 0.15), int(map_height * 0.25))
        print(f"Creando caverna central en ({cx},{cy}) con radios ({rx},{ry})")
        self._draw_ellipse(cx, cy, rx, ry, grid, config, 1)

class MediumHorizontalCavernCarver(ClearingCarver):
    def carve(self, leaf, grid, config):
        params = config['ellipse_params']
        min_rx_factor = 0.3
        max_rx_factor = 0.6
        stretch_factor = params.get('medium_stretch_factor', 2.5)
        max_possible_rx = leaf.width // 2 - 1
        rx_candidate = random.randint(int(leaf.width * min_rx_factor), int(leaf.width * max_rx_factor))
        rx = min(rx_candidate, max_possible_rx)
        if rx < params['min_radius'] or rx < 2: return
        ry = max(params['min_radius'], int(rx / stretch_factor))
        max_possible_ry = leaf.height // 2 - 1
        if ry > max_possible_ry: return
        ry = min(ry, max_possible_ry)
        start_cx, end_cx = leaf.x + rx, leaf.x + leaf.width - rx - 1
        if start_cx > end_cx: return
        start_cy, end_cy = leaf.y + ry, leaf.y + leaf.height - ry - 1
        if start_cy > end_cy: return
        cx, cy = random.randint(start_cx, end_cx), random.randint(start_cy, end_cy)
        self._draw_ellipse(cx, cy, rx, ry, grid, config, 1)

class CircleCarver(BaseEllipseCarver):
    def _calculate_radii(self, leaf, params):
        min_r, max_r = params['min_radius'], params['max_radius']
        max_possible_r = min(leaf.width // 2 - 1, leaf.height // 2 - 1)
        if max_possible_r < min_r: return None, None
        actual_max_r = min(max_r, max_possible_r)
        if min_r > actual_max_r: return None, None
        r = random.randint(min_r, actual_max_r)
        return r, r

class HorizontalEllipseCarver(BaseEllipseCarver):
    def _calculate_radii(self, leaf, params):
        min_r, max_r, stretch = params['min_radius'], params['max_radius'], params['stretch_factor']
        max_possible_rx = leaf.width // 2 - 1
        if max_possible_rx < min_r: return None, None
        actual_max_rx = min(max_r, max_possible_rx)
        if min_r > actual_max_rx: return None, None
        rx = random.randint(min_r, actual_max_rx)
        ry = max(1, int(rx / stretch))
        if (ry * 2) >= leaf.height -1:
            return None, None
        return rx, ry

class VerticalEllipseCarver(BaseEllipseCarver):
    def _calculate_radii(self, leaf, params):
        min_r, max_r, stretch = params['min_radius'], params['max_radius'], params['stretch_factor']
        max_possible_rx = leaf.width // 2 - 1
        max_possible_ry = leaf.height // 2 - 1
        if max_possible_rx < min_r: return None, None
        actual_max_rx = min(max_r, max_possible_rx)
        if min_r > actual_max_rx: return None, None
        rx = random.randint(min_r, actual_max_rx)
        ry = min(int(rx * stretch), max_possible_ry)
        return rx, ry

class RandomWalkCarver(ClearingCarver):
    def carve(self, leaf, grid, config):
        params = config['random_walk_carver']
        steps, brush_r = params['steps'], params['brush_radius']
        x, y = random.randint(leaf.x + 1, leaf.x + leaf.width - 2), random.randint(leaf.y + 1, leaf.y + leaf.height - 2)
        for _ in range(steps):
            self._draw_ellipse(x, y, brush_r, brush_r, grid, config, 1)
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x, y = x + dx, y + dy
            x, y = max(leaf.x + 1, min(x, leaf.x + leaf.width - 2)), max(leaf.y + 1, min(y, leaf.y + leaf.height - 2))

class RectangleCarver(ClearingCarver):
    def carve(self, leaf, grid, config):
        params = config['rectangle_carver']
        num_shapes = random.randint(*params['num_shapes'])
        for _ in range(num_shapes):
            min_w, max_w, min_h, max_h = params['min_width'], params['max_width'], params['min_height'], params['max_height']
            if leaf.width-2 < min_w or leaf.height-2 < min_h: continue
            rect_w, rect_h = random.randint(min_w, min(max_w, leaf.width - 2)), random.randint(min_h, min(max_h, leaf.height - 2))
            rect_x = random.randint(leaf.x + 1, leaf.x + leaf.width - rect_w - 1)
            rect_y = random.randint(leaf.y + 1, leaf.y + leaf.height - rect_h - 1)
            grid[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = 1

class TriangleCarver(ClearingCarver):
    def carve(self, leaf, grid, config):
        params = config['triangle_carver']
        num_shapes = random.randint(*params['num_shapes'])
        for _ in range(num_shapes):
            p1 = (random.randint(leaf.x, leaf.x + leaf.width-1), random.randint(leaf.y, leaf.y + leaf.height-1))
            p2 = (random.randint(leaf.x, leaf.x + leaf.width-1), random.randint(leaf.y, leaf.y + leaf.height-1))
            p3 = (random.randint(leaf.x, leaf.x + leaf.width-1), random.randint(leaf.y, leaf.y + leaf.height-1))
            min_x, max_x = max(leaf.x, min(p1[0], p2[0], p3[0])), min(leaf.x + leaf.width, max(p1[0], p2[0], p3[0]))
            min_y, max_y = max(leaf.y, min(p1[1], p2[1], p3[1])), min(leaf.y + leaf.height, max(p1[1], p2[1], p3[1]))
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if self._is_point_in_triangle((x, y), p1, p2, p3): grid[y, x] = 1
    def _sign(self, p1, p2, p3): return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    def _is_point_in_triangle(self, pt, v1, v2, v3):
        d1, d2, d3 = self._sign(pt, v1, v2), self._sign(pt, v2, v3), self._sign(pt, v3, v1)
        has_neg, has_pos = (d1 < 0) or (d2 < 0) or (d3 < 0), (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)


# --- 4. CLASES PARA COLOCAR OBSTÁCULOS (PARA CAMINOS) ---

class RockClusterPlacer(ObstaclePlacer):
    """Coloca rocas o grupos de rocas de diferentes formas y tamaños."""
    def place(self, grid, path_coords, config):
        map_height, map_width = grid.shape
        count = config.get('count', 1)
        min_r = config.get('min_radius', 1)
        max_r = config.get('max_radius', 2)
        stretch = config.get('stretch', 1.0)

        for _ in range(count):
            if not path_coords: continue
            center_y, center_x = random.choice(path_coords)
            ry = random.randint(min_r, max_r)
            rx = int(ry * stretch)
            
            for x_offset in range(-rx, rx + 1):
                for y_offset in range(-ry, ry + 1):
                    if (x_offset**2 / max(1, rx**2)) + (y_offset**2 / max(1, ry**2)) <= 1:
                        x, y = center_x + x_offset, center_y + y_offset
                        if 0 <= y < map_height and 0 <= x < map_width and grid[y, x] == 1:
                            grid[y, x] = 2

class LogPlacer(ObstaclePlacer):
    """Coloca obstáculos rectangulares largos, como troncos."""
    def place(self, grid, path_coords, config):
        map_height, map_width = grid.shape
        count = config.get('count', 1)
        min_len = config.get('min_length', 3)
        max_len = config.get('max_length', 6)
        thickness = config.get('thickness', 1)

        for _ in range(count):
            if not path_coords: continue
            start_y, start_x = random.choice(path_coords)
            length = random.randint(min_len, max_len)
            horizontal = random.choice([True, False])
            for i in range(length):
                for t in range(thickness):
                    if horizontal:
                        x, y = start_x + i, start_y + t
                    else:
                        x, y = start_x + t, start_y + i
                    if 0 <= y < map_height and 0 <= x < map_width and grid[y, x] == 1:
                        grid[y, x] = 2

# --- 5. ESTRATEGIAS DE GENERACIÓN PRINCIPALES ---

class BspDungeonStrategy(MapGenerationStrategy):
    class Leaf:
        def __init__(self, x, y, width, height):
            self.x, self.y, self.width, self.height = x, y, width, height
            self.child_1, self.child_2, self.room = None, None, None
        def split(self, min_leaf_size):
            if self.child_1 is not None or self.child_2 is not None: return False
            split_horizontally = random.choice([True, False])
            if self.width > self.height and self.width / self.height >= 1.25: split_horizontally = False
            elif self.height > self.width and self.height / self.width >= 1.25: split_horizontally = True
            max_size = (self.height if split_horizontally else self.width) - min_leaf_size
            if max_size <= min_leaf_size: return False
            split_point = random.randint(min_leaf_size, max_size)
            if split_horizontally:
                self.child_1 = BspDungeonStrategy.Leaf(self.x, self.y, self.width, split_point)
                self.child_2 = BspDungeonStrategy.Leaf(self.x, self.y + split_point, self.width, self.height - split_point)
            else:
                self.child_1 = BspDungeonStrategy.Leaf(self.x, self.y, split_point, self.height)
                self.child_2 = BspDungeonStrategy.Leaf(self.x + split_point, self.y, self.width - split_point, self.height)
            return True
        def create_room(self, config):
            if self.child_1 is not None or self.child_2 is not None: return
            padding, room_min_size = 2, config['room_min_size']
            if self.width < room_min_size + padding or self.height < room_min_size + padding: self.room = None; return
            shape_bias = random.choice(config['room_shape_biases'])
            try:
                if shape_bias == 'wide': min_w, max_w, min_h, max_h = int(self.width * 0.7), self.width - padding, room_min_size, int(self.height * 0.6)
                elif shape_bias == 'tall': min_w, max_w, min_h, max_h = room_min_size, int(self.width * 0.6), int(self.height * 0.7), self.height - padding
                else: min_w, max_w, min_h, max_h = room_min_size, self.width - padding, room_min_size, self.height - padding
                actual_min_w, actual_max_w = max(room_min_size, min_w), max(max(room_min_size, min_w), max_w)
                actual_min_h, actual_max_h = max(room_min_size, min_h), max(max(room_min_size, min_h), max_h)
                room_width, room_height = random.randint(actual_min_w, actual_max_w), random.randint(actual_min_h, actual_max_h)
                room_x, room_y = random.randint(self.x + 1, self.x + self.width - room_width - 1), random.randint(self.y + 1, self.y + self.height - room_height - 1)
                self.room = {'x': room_x, 'y': room_y, 'width': room_width, 'height': room_height}
            except ValueError: self.room = None
    def generate(self, config):
        grid = np.zeros((config['map_height'], config['map_width']), dtype=int)
        leaves = []
        map_w, map_h, min_leaf = config['map_width'], config['map_height'], config['min_leaf_size']
        root_leaf = self.Leaf(0, 0, map_w, map_h)
        leaves.append(root_leaf)
        did_split = True
        while did_split:
            did_split = False
            for leaf in list(leaves):
                if leaf.child_1 is None and leaf.child_2 is None and (leaf.width > min_leaf * 2 or leaf.height > min_leaf * 2):
                    if leaf.split(min_leaf): leaves.extend([leaf.child_1, leaf.child_2]); did_split = True
        final_leaves = [leaf for leaf in leaves if leaf.child_1 is None and leaf.child_2 is None]
        for leaf in final_leaves:
            leaf.create_room(config)
            if leaf.room: r = leaf.room; grid[r['y']:r['y']+r['height'], r['x']:r['x']+r['width']] = 1
        for parent in [leaf for leaf in leaves if leaf.child_1 is not None]:
            r1, r2 = self._get_room_from_branch(parent.child_1), self._get_room_from_branch(parent.child_2)
            if r1 and r2: self._connect_rooms(r1, r2, grid)
        return grid
    def _connect_rooms(self, r1, r2, grid):
        c1_x, c1_y = r1['x'] + r1['width'] // 2, r1['y'] + r1['height'] // 2
        c2_x, c2_y = r2['x'] + r2['width'] // 2, r2['y'] + r2['height'] // 2
        if random.choice([True, False]):
            for x in range(min(c1_x, c2_x), max(c1_x, c2_x) + 1): grid[c1_y, x] = 1
            for y in range(min(c1_y, c2_y), max(c1_y, c2_y) + 1): grid[y, c2_x] = 1
        else:
            for y in range(min(c1_y, c2_y), max(c1_y, c2_y) + 1): grid[y, c1_x] = 1
            for x in range(min(c1_x, c2_x), max(c1_x, c2_x) + 1): grid[c2_y, x] = 1
    def _get_room_from_branch(self, leaf):
        if hasattr(leaf, 'room') and leaf.room: return leaf.room
        if leaf.child_1 is None and leaf.child_2 is None: return None
        r1 = self._get_room_from_branch(leaf.child_1) if leaf.child_1 else None
        r2 = self._get_room_from_branch(leaf.child_2) if leaf.child_2 else None
        if r1 and r2: return random.choice([r1, r2])
        return r1 or r2

class HybridForestStrategy(MapGenerationStrategy):
    Leaf = BspDungeonStrategy.Leaf

    def __init__(self):
        self.carver_map = {
            'circle': CircleCarver(),
            'horizontal_ellipse': HorizontalEllipseCarver(),
            'vertical_ellipse': VerticalEllipseCarver(),
            'random_walk': RandomWalkCarver(),
            'rectangle': RectangleCarver(),
            'triangle': TriangleCarver(),
            'medium_horizontal_cavern': MediumHorizontalCavernCarver()
        }
        self.large_cavern_carver = LargeHorizontalCavernCarver()

    def generate(self, config):
        self.config = config
        self.grid = np.zeros((config['map_height'], config['map_width']), dtype=int)
        
        carver_choices = []
        for c in config['clearing_carvers']:
            if c['type'] in self.carver_map:
                carver_choices.extend([self.carver_map[c['type']]] * c['weight'])
            else:
                print(f"Advertencia: Tallador '{c['type']}' no encontrado.")

        leaves = self._partition_space()
        
        self.clearing_centers = []
        for leaf in [l for l in leaves if l.child_1 is None and l.child_2 is None]:
            if carver_choices:
                chosen_carver = random.choice(carver_choices)
                chosen_carver.carve(leaf, self.grid, self.config)
            self._apply_cellular_automata(leaf)
            
            floor_points = np.argwhere(self.grid[leaf.y:leaf.y+leaf.height, leaf.x:leaf.x+leaf.width] == 1)
            if floor_points.size > 0:
                local_center = random.choice(floor_points)
                leaf.center_point = (leaf.x + local_center[1], leaf.y + local_center[0])
                self.clearing_centers.append(leaf.center_point)
            else:
                leaf.center_point = None

        for parent in [l for l in leaves if l.child_1 is not None]:
            p1 = self._get_point_from_leaf(parent.child_1)
            p2 = self._get_point_from_leaf(parent.child_2)
            if p1 and p2: self._create_path_segment(p1, p2)
        
        self._add_extra_connections()
        self._add_forest_floor()
        self._create_entrances_and_exits()
        
        return self.grid

    def _partition_space(self):
        map_w, map_h, min_leaf = self.config['map_width'], self.config['map_height'], self.config['min_leaf_size']
        root_leaf = self.Leaf(0, 0, map_w, map_h)
        leaves = [root_leaf]
        did_split = True
        while did_split:
            did_split = False
            for leaf in list(leaves):
                if leaf.child_1 is None and leaf.child_2 is None and (leaf.width > min_leaf * 2 or leaf.height > min_leaf * 2):
                    if leaf.split(min_leaf):
                        leaves.extend([leaf.child_1, leaf.child_2])
                        did_split = True
        return leaves
        
    def _apply_cellular_automata(self, leaf):
        passes, birth_limit, death_limit = self.config['cellular_automata_passes'], self.config['cellular_automata_birth_limit'], self.config['cellular_automata_death_limit']
        temp_grid = self.grid.copy()
        for _ in range(passes):
            new_grid_segment = self.grid[leaf.y : leaf.y + leaf.height, leaf.x : leaf.x + leaf.width].copy()
            for y_local in range(leaf.height):
                for x_local in range(leaf.width):
                    global_x, global_y = leaf.x + x_local, leaf.y + y_local
                    if 0 < global_x < self.config['map_width'] -1 and 0 < global_y < self.config['map_height'] - 1:
                        neighbors = sum(1 for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dx == 0 and dy == 0) and 0 <= global_y + dy < self.config['map_height'] and 0 <= global_x + dx < self.config['map_width'] and temp_grid[global_y + dy, global_x + dx] == 1)
                        if temp_grid[global_y, global_x] == 1:
                            if neighbors < death_limit: new_grid_segment[y_local, x_local] = 0
                        elif neighbors > birth_limit: new_grid_segment[y_local, x_local] = 1
            self.grid[leaf.y : leaf.y + leaf.height, leaf.x : leaf.x + leaf.width] = new_grid_segment
            temp_grid[leaf.y : leaf.y + leaf.height, leaf.x : leaf.x + leaf.width] = new_grid_segment
            
    def _get_point_from_leaf(self, leaf):
        if hasattr(leaf, 'center_point') and leaf.center_point: return leaf.center_point
        if leaf.child_1 is None and leaf.child_2 is None: return None
        p1 = self._get_point_from_leaf(leaf.child_1) if leaf.child_1 else None
        p2 = self._get_point_from_leaf(leaf.child_2) if leaf.child_2 else None
        if p1 and p2: return random.choice([p1, p2])
        return p1 or p2

    def _create_path_segment(self, start_point, end_point):
        path = self._find_path_a_star(start_point, end_point)
        if not path: return
        path_width = self.config['path_width']
        radius = max(1, path_width // 2)
        carver = self.carver_map['circle']
        for x, y in path:
            carver._draw_ellipse(x, y, radius, radius, self.grid, self.config, 1)

    def _add_extra_connections(self):
        extra_prob = self.config['extra_path_connections_prob']
        if extra_prob <= 0 or len(self.clearing_centers) < 2: return
        for p1 in self.clearing_centers:
            if random.random() < extra_prob:
                potential_targets = [p for p in self.clearing_centers if p != p1]
                if potential_targets:
                    potential_targets.sort(key=lambda p: (p[0]-p1[0])**2 + (p[1]-p1[1])**2)
                    self._create_path_segment(p1, potential_targets[0])

    def _find_path_a_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from, g_score, f_score = {}, {}, {}
        g_score[start] = 0
        f_score[start] = self._heuristic(start, end)
        wiggle_room = self.config['path_wiggle_room']
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                path = []
                while current in came_from: path.append(current); current = came_from[current]
                return path[::-1]
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.config['map_width'] and 0 <= neighbor[1] < self.config['map_height']:
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1
                    cost_penalty = 0.1 if self.grid[neighbor[1], neighbor[0]] == 1 else 0
                    cost_penalty += random.uniform(0, wiggle_room)
                    tentative_g_score = g_score.get(current, float('inf')) + move_cost + cost_penalty
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor], g_score[neighbor] = current, tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
        
    def _heuristic(self, a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _add_forest_floor(self):
        density = self.config['forest_density']
        for y in range(self.config['map_height']):
            for x in range(self.config['map_width']):
                if self.grid[y, x] == 0 and random.random() < density: self.grid[y, x] = 1

    def _create_entrances_and_exits(self):
        num_entrances, entrance_width = self.config.get('num_entrances', 0), self.config.get('entrance_width', 1)
        if num_entrances == 0: return
        width, height = self.config['map_width'], self.config['map_height']
        possible_entry_points = []
        for x in range(1, width - 1):
            if self.grid[1, x] == 1: possible_entry_points.append((x, 0))
            if self.grid[height - 2, x] == 1: possible_entry_points.append((x, height - 1))
        for y in range(1, height - 1):
            if self.grid[y, 1] == 1: possible_entry_points.append((0, y))
            if self.grid[y, width - 2] == 1: possible_entry_points.append((width - 1, y))
        random.shuffle(possible_entry_points)
        for i, (center_x, center_y) in enumerate(possible_entry_points):
            if i >= num_entrances: break
            is_horizontal = (center_y == 0 or center_y == height - 1)
            for j in range(-(entrance_width // 2), (entrance_width // 2) + 1):
                x, y = (center_x + j, center_y) if is_horizontal else (center_x, center_y + j)
                if 0 <= x < width and 0 <= y < height: self.grid[y, x] = 1

class PathFocusedStrategy(MapGenerationStrategy):
    def __init__(self):
        self.obstacle_placers = {
            'rock_cluster': RockClusterPlacer(),
            'log': LogPlacer()
        }

    def generate(self, config):
        grid = np.zeros((config['map_height'], config['map_width']), dtype=int)
        
        print("Creando camino principal...")
        main_path_coords = self._create_winding_path(grid, config, is_main_path=True)
        self._carve_path(grid, main_path_coords, config['path_width'])

        print("Añadiendo bifurcaciones...")
        all_path_coords = set(main_path_coords)
        for _ in range(config['num_bifurcations']):
            if not all_path_coords: break
            start_node = random.choice(list(all_path_coords))
            bifurcation_coords = self._create_winding_path(grid, config, start_node=start_node)
            self._carve_path(grid, bifurcation_coords, config['bifurcation_width'])
            all_path_coords.update(bifurcation_coords)
        
        print("Colocando obstáculos...")
        path_pixels = list(zip(*np.where(grid == 1)))
        
        for obstacle_config in config.get('obstacles', []):
            obs_type = obstacle_config['type']
            if obs_type in self.obstacle_placers:
                placer = self.obstacle_placers[obs_type]
                print(f" - Colocando {obstacle_config['count']} de tipo '{obs_type}'")
                placer.place(grid, path_pixels, obstacle_config)
            else:
                print(f"Advertencia: Tipo de obstáculo '{obs_type}' no reconocido.")

        return grid

    def _carve_path(self, grid, path_nodes, width):
        radius = max(1, width // 2)
        for y, x in path_nodes:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            grid[ny, nx] = 1

    def _create_winding_path(self, grid, config, is_main_path=False, start_node=None):
        h, w = grid.shape
        if is_main_path:
            start_y = random.randint(h // 4, 3 * h // 4)
            end_y = random.randint(h // 4, 3 * h // 4)
            start = (start_y, 1)
            end = (end_y, w - 2)
            max_len = float('inf')
        elif start_node:
            start = start_node
            end = (random.randint(0, h-1), random.randint(0, w-1))
            max_len = random.randint(*config['bifurcation_length'])
        else:
            return []

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from, g_score = {}, {}
        g_score[start] = 0
        path_found = None
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if not is_main_path and g_score.get(current, 0) > max_len:
                path_found = current
                break
            if is_main_path and current == end:
                path_found = current
                break

            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor = (current[0] + dy, current[1] + dx)
                if 1 <= neighbor[0] < h-1 and 1 <= neighbor[1] < w-1:
                    cost = 1 + random.uniform(0, config['path_wiggle_room'])
                    tentative_g_score = g_score.get(current, float('inf')) + cost
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + (abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1]))
                        heapq.heappush(open_set, (f_score, neighbor))
        
        if path_found:
            path = []
            curr = path_found
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            return path[::-1]
        return []

# --- 6. CLASE PRINCIPAL DEL GENERADOR (Contexto) ---
class MapGenerator:
    def __init__(self, environment_type='dungeon'):
        if environment_type not in ENVIRONMENT_PRESETS:
            raise ValueError(f"Tipo de entorno no válido: '{environment_type}'")
        self.config = ENVIRONMENT_PRESETS[environment_type]
        
        algorithm_name = self.config.get('generator_algorithm')
        if algorithm_name == 'bsp': self.strategy = BspDungeonStrategy()
        elif algorithm_name == 'hybrid_forest': self.strategy = HybridForestStrategy()
        elif algorithm_name == 'path_focused': self.strategy = PathFocusedStrategy()
        else: raise ValueError(f"Algoritmo de generación desconocido: {algorithm_name}")

    def generate(self):
        return self.strategy.generate(self.config)

# --- 7. FUNCIÓN PARA IMPRIMIR EL MAPA ---
def print_map(grid, environment_type):
    if 'path_focused' in environment_type:
        chars = {0: '♣', 1: '░', 2: '█'} # 0=Muro, 1=Camino, 2=Obstáculo
    elif 'forest' in environment_type:
        chars = {0: '♣', 1: '.'}
    else: #dungeon
        chars = {0: '#', 1: '.'}
        
    print(f"--- Mapa generado: {environment_type.replace('_', ' ').title()} ---")
    for row in grid:
        print("".join([chars.get(cell, '?') for cell in row]))
    print("\n")

# --- 8. BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("Generando el nuevo mapa de tipo 'Camino con Obstáculos'...")
    path_generator = MapGenerator(environment_type='path_focused')
    path_map = path_generator.generate()
    print_map(path_map, 'path_focused')

    print("Generando un bosque con la estrategia Híbrida...")
    forest_generator = MapGenerator(environment_type='forest')
    forest_map = forest_generator.generate()
    print_map(forest_map, 'forest')
    
    print("Generando una mazmorra con la estrategia BSP clásica...")
    dungeon_generator = MapGenerator(environment_type='dungeon')
    dungeon_map = dungeon_generator.generate()
    print_map(dungeon_map, 'dungeon')