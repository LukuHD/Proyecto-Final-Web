"""
Map generation strategies module.

Contains all strategy classes for generating different types of maps.
"""

import random
import math
import heapq
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

from core.carvers import ClearingCarver, CircleCarver
from core.obstacles import ObstaclePlacer


class MapGenerationStrategy(ABC):
    """Abstract base class for map generation strategies."""
    
    @abstractmethod
    def generate(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate a map based on the configuration."""
        pass
    
    def get_name(self) -> str:
        """Return the name of this strategy."""
        return self.__class__.__name__
    
    def get_supported_features(self) -> List[str]:
        """Return list of features supported by this strategy."""
        return ["basic_generation"]


class BspDungeonStrategy(MapGenerationStrategy):
    """BSP (Binary Space Partition) dungeon generation strategy."""
    
    def get_name(self) -> str:
        return "BSP Dungeon Generator"
    
    def get_supported_features(self) -> List[str]:
        return ["rooms", "corridors", "bsp_tree"]
    
    class Leaf:
        """A leaf node in the BSP tree."""
        
        def __init__(self, x, y, width, height):
            self.x, self.y, self.width, self.height = x, y, width, height
            self.child_1, self.child_2, self.room = None, None, None
        
        def split(self, min_leaf_size):
            if self.child_1 is not None or self.child_2 is not None: 
                return False
            split_horizontally = random.choice([True, False])
            if self.width > self.height and self.width / self.height >= 1.25: 
                split_horizontally = False
            elif self.height > self.width and self.height / self.width >= 1.25: 
                split_horizontally = True
            max_size = (self.height if split_horizontally else self.width) - min_leaf_size
            if max_size <= min_leaf_size: 
                return False
            split_point = random.randint(min_leaf_size, max_size)
            if split_horizontally:
                self.child_1 = BspDungeonStrategy.Leaf(self.x, self.y, self.width, split_point)
                self.child_2 = BspDungeonStrategy.Leaf(self.x, self.y + split_point, self.width, self.height - split_point)
            else:
                self.child_1 = BspDungeonStrategy.Leaf(self.x, self.y, split_point, self.height)
                self.child_2 = BspDungeonStrategy.Leaf(self.x + split_point, self.y, self.width - split_point, self.height)
            return True
        
        def create_room(self, config):
            if self.child_1 is not None or self.child_2 is not None: 
                return
            room_min_size = config.get('room_min_size', 4)
            padding = max(0, config.get('room_padding', 2))
            if self.width < room_min_size + padding or self.height < room_min_size + padding: 
                self.room = None
                return
            shape_bias = random.choice(config['room_shape_biases'])
            try:
                if shape_bias == 'wide': 
                    min_w, max_w = int(self.width * 0.7), self.width - padding
                    min_h, max_h = room_min_size, int(self.height * 0.6)
                elif shape_bias == 'tall': 
                    min_w, max_w = room_min_size, int(self.width * 0.6)
                    min_h, max_h = int(self.height * 0.7), self.height - padding
                else: 
                    min_w, max_w = room_min_size, self.width - padding
                    min_h, max_h = room_min_size, self.height - padding
                actual_min_w = max(room_min_size, min_w)
                actual_max_w = max(actual_min_w, max_w)
                actual_min_h = max(room_min_size, min_h)
                actual_max_h = max(actual_min_h, max_h)
                room_width = random.randint(actual_min_w, actual_max_w)
                room_height = random.randint(actual_min_h, actual_max_h)
                room_x = random.randint(self.x + 1, self.x + self.width - room_width - 1)
                room_y = random.randint(self.y + 1, self.y + self.height - room_height - 1)
                self.room = {'x': room_x, 'y': room_y, 'width': room_width, 'height': room_height}
            except ValueError: 
                self.room = None
    
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
                    if leaf.split(min_leaf): 
                        leaves.extend([leaf.child_1, leaf.child_2])
                        did_split = True
        final_leaves = [leaf for leaf in leaves if leaf.child_1 is None and leaf.child_2 is None]
        for leaf in final_leaves:
            leaf.create_room(config)
            if leaf.room: 
                r = leaf.room
                grid[r['y']:r['y'] + r['height'], r['x']:r['x'] + r['width']] = 1
        for parent in [leaf for leaf in leaves if leaf.child_1 is not None]:
            r1 = self._get_room_from_branch(parent.child_1)
            r2 = self._get_room_from_branch(parent.child_2)
            if r1 and r2: 
                self._connect_rooms(r1, r2, grid)
        return grid
    
    def _connect_rooms(self, r1, r2, grid):
        c1_x = r1['x'] + r1['width'] // 2
        c1_y = r1['y'] + r1['height'] // 2
        c2_x = r2['x'] + r2['width'] // 2
        c2_y = r2['y'] + r2['height'] // 2
        if random.choice([True, False]):
            for x in range(min(c1_x, c2_x), max(c1_x, c2_x) + 1): 
                grid[c1_y, x] = 1
            for y in range(min(c1_y, c2_y), max(c1_y, c2_y) + 1): 
                grid[y, c2_x] = 1
        else:
            for y in range(min(c1_y, c2_y), max(c1_y, c2_y) + 1): 
                grid[y, c1_x] = 1
            for x in range(min(c1_x, c2_x), max(c1_x, c2_x) + 1): 
                grid[c2_y, x] = 1
    
    def _get_room_from_branch(self, leaf):
        if hasattr(leaf, 'room') and leaf.room: 
            return leaf.room
        if leaf.child_1 is None and leaf.child_2 is None: 
            return None
        r1 = self._get_room_from_branch(leaf.child_1) if leaf.child_1 else None
        r2 = self._get_room_from_branch(leaf.child_2) if leaf.child_2 else None
        if r1 and r2: 
            return random.choice([r1, r2])
        return r1 or r2


class HybridForestStrategy(MapGenerationStrategy):
    """Hybrid forest generation strategy with clearings and paths."""
    
    def get_name(self) -> str:
        return "Hybrid Forest Generator"
    
    def get_supported_features(self) -> List[str]:
        features = ["clearings", "paths", "cellular_automata", "entrances"]
        if hasattr(self, 'obstacle_placers') and self.obstacle_placers:
            features.append("obstacles")
        return features
    
    def __init__(self, carver_map: Dict[str, ClearingCarver], obstacle_placers: Dict[str, ObstaclePlacer] = None):
        self.carver_map = carver_map
        self.obstacle_placers = obstacle_placers or {}

    class Leaf:
        """A leaf node for forest partitioning."""
        
        def __init__(self, x, y, width, height):
            self.x, self.y, self.width, self.height = x, y, width, height
            self.child_1, self.child_2, self.room = None, None, None
            self.center_point = None
            
        def split(self, min_leaf_size):
            if self.child_1 is not None or self.child_2 is not None: 
                return False
            split_horizontally = random.choice([True, False])
            if self.width > self.height and self.width / self.height >= 1.25: 
                split_horizontally = False
            elif self.height > self.width and self.height / self.width >= 1.25: 
                split_horizontally = True
            max_size = (self.height if split_horizontally else self.width) - min_leaf_size
            if max_size <= min_leaf_size: 
                return False
            split_point = random.randint(min_leaf_size, max_size)
            if split_horizontally:
                self.child_1 = HybridForestStrategy.Leaf(self.x, self.y, self.width, split_point)
                self.child_2 = HybridForestStrategy.Leaf(self.x, self.y + split_point, self.width, self.height - split_point)
            else:
                self.child_1 = HybridForestStrategy.Leaf(self.x, self.y, split_point, self.height)
                self.child_2 = HybridForestStrategy.Leaf(self.x + split_point, self.y, self.width - split_point, self.height)
            return True

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
            
            floor_points = np.argwhere(self.grid[leaf.y:leaf.y + leaf.height, leaf.x:leaf.x + leaf.width] == 1)
            if floor_points.size > 0:
                local_center = random.choice(floor_points)
                leaf.center_point = (leaf.x + local_center[1], leaf.y + local_center[0])
                self.clearing_centers.append(leaf.center_point)
            else:
                leaf.center_point = None

        for parent in [l for l in leaves if l.child_1 is not None]:
            p1 = self._get_point_from_leaf(parent.child_1)
            p2 = self._get_point_from_leaf(parent.child_2)
            if p1 and p2: 
                self._create_path_segment(p1, p2)
        
        self._add_extra_connections()
        
        # AÑADIR OBSTÁCULOS AL BOSQUE
        if self.obstacle_placers and 'obstacles' in config:
            print("Colocando obstáculos en el bosque...")
            path_and_clearing_coords = list(zip(*np.where(self.grid == 1)))
            
            for obstacle_config in config.get('obstacles', []):
                obs_type = obstacle_config['type']
                if obs_type in self.obstacle_placers:
                    placer = self.obstacle_placers[obs_type]
                    print(f" - Colocando {obstacle_config['count']} de tipo '{obs_type}'")
                    placer.place(self.grid, path_and_clearing_coords, obstacle_config)
                else:
                    print(f"Advertencia: Tipo de obstáculo '{obs_type}' no reconocido.")
        
        self._add_forest_floor()
        self._create_entrances_and_exits()
        
        return self.grid

    def _partition_space(self):
        map_w = self.config['map_width']
        map_h = self.config['map_height']
        min_leaf = self.config['min_leaf_size']
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
        passes = self.config['cellular_automata_passes']
        birth_limit = self.config['cellular_automata_birth_limit']
        death_limit = self.config['cellular_automata_death_limit']
        temp_grid = self.grid.copy()
        for _ in range(passes):
            new_grid_segment = self.grid[leaf.y:leaf.y + leaf.height, leaf.x:leaf.x + leaf.width].copy()
            for y_local in range(leaf.height):
                for x_local in range(leaf.width):
                    global_x, global_y = leaf.x + x_local, leaf.y + y_local
                    if 0 < global_x < self.config['map_width'] - 1 and 0 < global_y < self.config['map_height'] - 1:
                        neighbors = sum(1 for dy in [-1, 0, 1] for dx in [-1, 0, 1] 
                                       if not (dx == 0 and dy == 0) 
                                       and 0 <= global_y + dy < self.config['map_height'] 
                                       and 0 <= global_x + dx < self.config['map_width'] 
                                       and temp_grid[global_y + dy, global_x + dx] == 1)
                        if temp_grid[global_y, global_x] == 1:
                            if neighbors < death_limit: 
                                new_grid_segment[y_local, x_local] = 0
                        elif neighbors > birth_limit: 
                            new_grid_segment[y_local, x_local] = 1
            self.grid[leaf.y:leaf.y + leaf.height, leaf.x:leaf.x + leaf.width] = new_grid_segment
            temp_grid[leaf.y:leaf.y + leaf.height, leaf.x:leaf.x + leaf.width] = new_grid_segment
            
    def _get_point_from_leaf(self, leaf):
        if hasattr(leaf, 'center_point') and leaf.center_point: 
            return leaf.center_point
        if leaf.child_1 is None and leaf.child_2 is None: 
            return None
        p1 = self._get_point_from_leaf(leaf.child_1) if leaf.child_1 else None
        p2 = self._get_point_from_leaf(leaf.child_2) if leaf.child_2 else None
        if p1 and p2: 
            return random.choice([p1, p2])
        return p1 or p2

    def _create_path_segment(self, start_point, end_point):
        path = self._find_path_a_star(start_point, end_point)
        if not path: 
            return
        path_width = self.config['path_width']
        radius = max(1, path_width // 2)
        carver = self.carver_map['circle']
        for x, y in path:
            carver._draw_ellipse(x, y, radius, radius, self.grid, self.config, 1)

    def _add_extra_connections(self):
        extra_prob = self.config['extra_path_connections_prob']
        if extra_prob <= 0 or len(self.clearing_centers) < 2: 
            return
        for p1 in self.clearing_centers:
            if random.random() < extra_prob:
                potential_targets = [p for p in self.clearing_centers if p != p1]
                if potential_targets:
                    potential_targets.sort(key=lambda p: (p[0] - p1[0])**2 + (p[1] - p1[1])**2)
                    self._create_path_segment(p1, potential_targets[0])

    def _find_path_a_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {}
        f_score = {}
        g_score[start] = 0
        f_score[start] = self._heuristic(start, end)
        wiggle_room = self.config['path_wiggle_room']
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                path = []
                while current in came_from: 
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.config['map_width'] and 0 <= neighbor[1] < self.config['map_height']:
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1
                    cost_penalty = 0.1 if self.grid[neighbor[1], neighbor[0]] == 1 else 0
                    cost_penalty += random.uniform(0, wiggle_room)
                    tentative_g_score = g_score.get(current, float('inf')) + move_cost + cost_penalty
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
        
    def _heuristic(self, a, b): 
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _add_forest_floor(self):
        density = self.config['forest_density']
        for y in range(self.config['map_height']):
            for x in range(self.config['map_width']):
                if self.grid[y, x] == 0 and random.random() < density: 
                    self.grid[y, x] = 1

    def _create_entrances_and_exits(self):
        num_entrances = self.config.get('num_entrances', 0)
        entrance_width = self.config.get('entrance_width', 1)
        if num_entrances == 0: 
            return
        width, height = self.config['map_width'], self.config['map_height']
        possible_entry_points = []
        for x in range(1, width - 1):
            if self.grid[1, x] == 1: 
                possible_entry_points.append((x, 0))
            if self.grid[height - 2, x] == 1: 
                possible_entry_points.append((x, height - 1))
        for y in range(1, height - 1):
            if self.grid[y, 1] == 1: 
                possible_entry_points.append((0, y))
            if self.grid[y, width - 2] == 1: 
                possible_entry_points.append((width - 1, y))
        random.shuffle(possible_entry_points)
        for i, (center_x, center_y) in enumerate(possible_entry_points):
            if i >= num_entrances: 
                break
            is_horizontal = (center_y == 0 or center_y == height - 1)
            for j in range(-(entrance_width // 2), (entrance_width // 2) + 1):
                x, y = (center_x + j, center_y) if is_horizontal else (center_x, center_y + j)
                if 0 <= x < width and 0 <= y < height: 
                    self.grid[y, x] = 1


class EnhancedPathFocusedStrategy(MapGenerationStrategy):
    """Enhanced path-focused strategy with natural paths and central hub."""
    
    def get_name(self) -> str:
        return "Enhanced Path Focused Generator"
    
    def get_supported_features(self) -> List[str]:
        return ["main_path", "bifurcations", "obstacles", "natural_paths", "central_hub"]
    
    def __init__(self, obstacle_placers: Dict[str, ObstaclePlacer]):
        self.obstacle_placers = obstacle_placers

    def generate(self, config):
        grid = np.zeros((config['map_height'], config['map_width']), dtype=int)
        
        print("Creando camino principal con formas naturales...")
        
        # Generar hub central primero si está configurado
        hub_center = None
        if config.get('has_central_hub', False):
            hub_center = self._create_central_hub(grid, config)
        
        # Generar camino principal
        main_path_coords = self._create_natural_path(grid, config, is_main_path=True, hub_center=hub_center)
        main_path_width = config['path_width']
        self._carve_path(grid, main_path_coords, main_path_width)

        print("Añadiendo bifurcaciones naturales...")
        all_path_coords = set(main_path_coords)
        
        # Conectar hub central si existe
        if hub_center:
            hub_connections = self._connect_hub_to_paths(grid, config, hub_center, all_path_coords)
            all_path_coords.update(hub_connections)
        
        # Generar bifurcaciones
        for _ in range(config['num_bifurcations']):
            if not all_path_coords: 
                break
            start_node = random.choice(list(all_path_coords))
            bifurcation_coords = self._create_natural_path(grid, config, start_node=start_node, is_main_path=False)
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

    def _create_central_hub(self, grid, config) -> Tuple[int, int]:
        """Crea un hub central ovalado grande."""
        center_x = config['map_width'] // 2
        center_y = config['map_height'] // 2
        hub_width, hub_height = config.get('central_hub_size', (12, 8))
        
        print(f"Creando hub central en ({center_x}, {center_y}) de tamaño {hub_width}x{hub_height}")
        
        # Usar el método de ellipse carver para crear el hub
        carver = CircleCarver()
        carver._draw_ellipse(center_x, center_y, hub_width, hub_height, grid, config, 1)
        
        return (center_y, center_x)

    def _connect_hub_to_paths(self, grid, config, hub_center, existing_paths):
        """Conecta el hub central a los caminos existentes."""
        hub_y, hub_x = hub_center
        connection_paths = config.get('hub_connection_paths', 3)
        path_width = config.get('hub_path_width', config['path_width'])
        connected_points = []
        
        print(f"Conectando hub a {connection_paths} caminos...")
        
        for i in range(connection_paths):
            # Encontrar un punto en el borde del hub para conectar
            angle = (2 * math.pi * i) / connection_paths
            connect_x = hub_x + int(config['central_hub_size'][0] * math.cos(angle))
            connect_y = hub_y + int(config['central_hub_size'][1] * math.sin(angle))
            
            # Encontrar el camino más cercano para conectar
            if existing_paths:
                closest_point = min(existing_paths, 
                                  key=lambda p: (p[0] - connect_y)**2 + (p[1] - connect_x)**2)
                
                # Crear camino de conexión natural
                connection_path = self._create_natural_segment(
                    (connect_y, connect_x), closest_point, config
                )
                self._carve_path(grid, connection_path, path_width)
                connected_points.extend(connection_path)
        
        return connected_points

    def _create_natural_path(self, grid, config, is_main_path=False, start_node=None, hub_center=None):
        """Crea caminos con formas naturales: rectos, curvados suaves, o en S."""
        h, w = grid.shape
        
        if is_main_path:
            # Para camino principal, conectar bordes opuestos
            if hub_center and random.random() < 0.7:  # 70% de probabilidad de pasar por el hub
                # Crear camino que pase por el hub
                start_side = random.choice(['left', 'right', 'top', 'bottom'])
                end_side = random.choice(['left', 'right', 'top', 'bottom'])
                while end_side == start_side:
                    end_side = random.choice(['left', 'right', 'top', 'bottom'])
                
                start_point = self._get_border_point(w, h, start_side)
                end_point = self._get_border_point(w, h, end_side)
                
                # Crear camino que pase por el hub
                path1 = self._create_natural_segment(start_point, (hub_center[0], hub_center[1]), config)
                path2 = self._create_natural_segment((hub_center[0], hub_center[1]), end_point, config)
                return path1 + path2[1:]
            else:
                # Camino normal de borde a borde
                start_side = random.choice(['left', 'right'])
                end_side = 'right' if start_side == 'left' else 'left'
                start_point = self._get_border_point(w, h, start_side)
                end_point = self._get_border_point(w, h, end_side)
        elif start_node:
            # Bifurcación desde un nodo existente
            start_point = start_node
            end_x = random.randint(w // 4, 3 * w // 4)
            end_y = random.randint(h // 4, 3 * h // 4)
            end_point = (end_y, end_x)
        else:
            return []

        return self._create_natural_segment(start_point, end_point, config)

    def _get_border_point(self, width, height, side):
        """Obtiene un punto en el borde del mapa."""
        if side == 'left':
            return (random.randint(1, height - 2), 1)
        elif side == 'right':
            return (random.randint(1, height - 2), width - 2)
        elif side == 'top':
            return (1, random.randint(1, width - 2))
        else:  # bottom
            return (height - 2, random.randint(1, width - 2))

    def _create_natural_segment(self, start, end, config):
        """Crea un segmento de camino con forma natural."""
        styles = config.get('path_styles', ['natural'])
        weights = config.get('path_style_weights', [1])
        
        style = random.choices(styles, weights=weights)[0]
        
        if style == 'straight':
            return self._create_straight_path(start, end, config)
        elif style == 'gentle_curve':
            return self._create_gentle_curve_path(start, end, config)
        elif style == 's_curve':
            return self._create_s_curve_path(start, end, config)
        else:  # 'natural'
            return self._create_natural_winding_path(start, end, config)

    def _create_straight_path(self, start, end, config):
        """Camino perfectamente recto."""
        path = []
        start_y, start_x = start
        end_y, end_x = end
        
        steps = max(abs(end_x - start_x), abs(end_y - start_y))
        if steps == 0:
            return [start]
            
        for i in range(steps + 1):
            t = i / steps
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            path.append((y, x))
            
        return path

    def _create_gentle_curve_path(self, start, end, config):
        """Camino con curva suave."""
        start_y, start_x = start
        end_y, end_x = end
        
        mid_x = (start_x + end_x) // 2
        mid_y = (start_y + end_y) // 2
        
        max_dev = config.get('max_curve_deviation', 0.3)
        dev_x = int((end_x - start_x) * max_dev * random.uniform(-1, 1))
        dev_y = int((end_y - start_y) * max_dev * random.uniform(-1, 1))
        
        control_x = mid_x + dev_x
        control_y = mid_y + dev_y
        
        return self._quadratic_bezier(start, (control_y, control_x), end, 20)

    def _create_s_curve_path(self, start, end, config):
        """Camino en forma de S suave."""
        start_y, start_x = start
        end_y, end_x = end
        
        third_x = start_x + (end_x - start_x) // 3
        two_thirds_x = start_x + 2 * (end_x - start_x) // 3
        
        max_dev = config.get('max_curve_deviation', 0.3)
        dev1 = int((end_y - start_y) * max_dev * random.uniform(0.5, 1.0))
        dev2 = int((end_y - start_y) * max_dev * random.uniform(0.5, 1.0))
        
        control1 = (start_y + dev1, third_x)
        control2 = (start_y - dev2, two_thirds_x)
        
        return self._cubic_bezier(start, control1, control2, end, 30)

    def _create_natural_winding_path(self, start, end, config):
        """Camino natural que combina segmentos rectos y curvos."""
        start_y, start_x = start
        end_y, end_x = end
        
        path = [start]
        current = start
        
        num_segments = random.randint(
            config.get('min_straight_segments', 3),
            config.get('max_straight_segments', 8)
        )
        
        for i in range(1, num_segments):
            t = i / num_segments
            target_x = int(start_x + t * (end_x - start_x))
            target_y = int(start_y + t * (end_y - start_y))
            
            variation = random.uniform(0.1, 0.3)
            var_x = int((end_x - start_x) * variation * random.uniform(-1, 1))
            var_y = int((end_y - start_y) * variation * random.uniform(-1, 1))
            
            next_point = (target_y + var_y, target_x + var_x)
            
            if random.random() < 0.7:
                segment = self._create_straight_path(current, next_point, config)
            else:
                segment = self._create_gentle_curve_path(current, next_point, config)
            
            path.extend(segment[1:])
            current = next_point
        
        final_segment = self._create_straight_path(current, end, config)
        path.extend(final_segment[1:])
        
        return path

    def _quadratic_bezier(self, p0, p1, p2, steps):
        """Genera una curva cuadrática de Bézier."""
        path = []
        p0_y, p0_x = p0
        p1_y, p1_x = p1
        p2_y, p2_x = p2
        
        for i in range(steps + 1):
            t = i / steps
            x = int((1 - t)**2 * p0_x + 2 * (1 - t) * t * p1_x + t**2 * p2_x)
            y = int((1 - t)**2 * p0_y + 2 * (1 - t) * t * p1_y + t**2 * p2_y)
            path.append((y, x))
            
        return path

    def _cubic_bezier(self, p0, p1, p2, p3, steps):
        """Genera una curva cúbica de Bézier."""
        path = []
        p0_y, p0_x = p0
        p1_y, p1_x = p1
        p2_y, p2_x = p2
        p3_y, p3_x = p3
        
        for i in range(steps + 1):
            t = i / steps
            x = int((1 - t)**3 * p0_x + 3 * (1 - t)**2 * t * p1_x + 
                   3 * (1 - t) * t**2 * p2_x + t**3 * p3_x)
            y = int((1 - t)**3 * p0_y + 3 * (1 - t)**2 * t * p1_y + 
                   3 * (1 - t) * t**2 * p2_y + t**3 * p3_y)
            path.append((y, x))
            
        return path

    def _carve_path(self, grid, path_nodes, path_width):
        """Tallar el camino en el grid con el ancho especificado."""
        radius = max(1, path_width // 2)
        for y, x in path_nodes:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            grid[ny, nx] = 1


class PathFocusedStrategy(MapGenerationStrategy):
    """Basic path-focused generation strategy."""
    
    def get_name(self) -> str:
        return "Path Focused Generator"
    
    def get_supported_features(self) -> List[str]:
        return ["main_path", "bifurcations", "obstacles"]
    
    def __init__(self, obstacle_placers: Dict[str, ObstaclePlacer]):
        self.obstacle_placers = obstacle_placers

    def generate(self, config):
        grid = np.zeros((config['map_height'], config['map_width']), dtype=int)
        
        print("Creando camino principal...")
        main_path_coords = self._create_winding_path(grid, config, is_main_path=True)
        self._carve_path(grid, main_path_coords, config['path_width'])

        print("Añadiendo bifurcaciones...")
        all_path_coords = set(main_path_coords)
        for _ in range(config['num_bifurcations']):
            if not all_path_coords: 
                break
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
            end = (random.randint(0, h - 1), random.randint(0, w - 1))
            max_len = random.randint(*config['bifurcation_length'])
        else:
            return []

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {}
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

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dy, current[1] + dx)
                if 1 <= neighbor[0] < h - 1 and 1 <= neighbor[1] < w - 1:
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


__all__ = [
    'MapGenerationStrategy',
    'BspDungeonStrategy',
    'HybridForestStrategy',
    'EnhancedPathFocusedStrategy',
    'PathFocusedStrategy',
]
