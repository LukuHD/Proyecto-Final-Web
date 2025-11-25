"""
Clearing carvers module.

Contains all carver classes for creating clearings, shapes, and open spaces in maps.
"""

import random
import math
import numpy as np
from abc import ABC, abstractmethod


class ClearingCarver(ABC):
    """Abstract base class for clearing carvers."""
    
    @abstractmethod
    def carve(self, leaf, grid, config): 
        """Carve a clearing into the grid."""
        pass

    def _draw_ellipse(self, cx, cy, rx, ry, grid, config, value):
        """Draw an ellipse at the specified position."""
        if rx <= 0: rx = 1
        if ry <= 0: ry = 1
        for x_offset in range(-rx, rx + 1):
            for y_offset in range(-ry, ry + 1):
                x, y = cx + x_offset, cy + y_offset
                if 0 <= x < config['map_width'] and 0 <= y < config['map_height']:
                    if (x_offset**2 / rx**2) + (y_offset**2 / ry**2) <= 1:
                        grid[y, x] = value


class BaseEllipseCarver(ClearingCarver):
    """Base class for ellipse-based carvers."""
    
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
        """Calculate the radii for the ellipse."""
        pass


class CircleCarver(BaseEllipseCarver):
    """Carver that creates circular clearings."""
    
    def _calculate_radii(self, leaf, params):
        min_r, max_r = params['min_radius'], params['max_radius']
        max_possible_r = min(leaf.width // 2 - 1, leaf.height // 2 - 1)
        if max_possible_r < min_r: return None, None
        actual_max_r = min(max_r, max_possible_r)
        if min_r > actual_max_r: return None, None
        r = random.randint(min_r, actual_max_r)
        return r, r


class HorizontalEllipseCarver(BaseEllipseCarver):
    """Carver that creates horizontally-stretched elliptical clearings."""
    
    def _calculate_radii(self, leaf, params):
        min_r, max_r, stretch = params['min_radius'], params['max_radius'], params['stretch_factor']
        max_possible_rx = leaf.width // 2 - 1
        if max_possible_rx < min_r: return None, None
        actual_max_rx = min(max_r, max_possible_rx)
        if min_r > actual_max_rx: return None, None
        rx = random.randint(min_r, actual_max_rx)
        ry = max(1, int(rx / stretch))
        if (ry * 2) >= leaf.height - 1:
            return None, None
        return rx, ry


class VerticalEllipseCarver(BaseEllipseCarver):
    """Carver that creates vertically-stretched elliptical clearings."""
    
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


class MediumHorizontalCavernCarver(ClearingCarver):
    """Carver that creates medium-sized horizontal cavern clearings."""
    
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


class RandomWalkCarver(ClearingCarver):
    """Carver that creates clearings using random walk algorithm."""
    
    def carve(self, leaf, grid, config):
        params = config['random_walk_carver']
        steps, brush_r = params['steps'], params['brush_radius']
        x = random.randint(leaf.x + 1, leaf.x + leaf.width - 2)
        y = random.randint(leaf.y + 1, leaf.y + leaf.height - 2)
        for _ in range(steps):
            self._draw_ellipse(x, y, brush_r, brush_r, grid, config, 1)
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x, y = x + dx, y + dy
            x = max(leaf.x + 1, min(x, leaf.x + leaf.width - 2))
            y = max(leaf.y + 1, min(y, leaf.y + leaf.height - 2))


class RectangleCarver(ClearingCarver):
    """Carver that creates rectangular clearings."""
    
    def carve(self, leaf, grid, config):
        params = config['rectangle_carver']
        num_shapes = random.randint(*params['num_shapes'])
        for _ in range(num_shapes):
            min_w, max_w = params['min_width'], params['max_width']
            min_h, max_h = params['min_height'], params['max_height']
            if leaf.width - 2 < min_w or leaf.height - 2 < min_h: continue
            rect_w = random.randint(min_w, min(max_w, leaf.width - 2))
            rect_h = random.randint(min_h, min(max_h, leaf.height - 2))
            rect_x = random.randint(leaf.x + 1, leaf.x + leaf.width - rect_w - 1)
            rect_y = random.randint(leaf.y + 1, leaf.y + leaf.height - rect_h - 1)
            grid[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 1


class TriangleCarver(ClearingCarver):
    """Carver that creates triangular clearings."""
    
    def carve(self, leaf, grid, config):
        params = config['triangle_carver']
        num_shapes = random.randint(*params['num_shapes'])
        for _ in range(num_shapes):
            p1 = (random.randint(leaf.x, leaf.x + leaf.width - 1), 
                  random.randint(leaf.y, leaf.y + leaf.height - 1))
            p2 = (random.randint(leaf.x, leaf.x + leaf.width - 1), 
                  random.randint(leaf.y, leaf.y + leaf.height - 1))
            p3 = (random.randint(leaf.x, leaf.x + leaf.width - 1), 
                  random.randint(leaf.y, leaf.y + leaf.height - 1))
            min_x = max(leaf.x, min(p1[0], p2[0], p3[0]))
            max_x = min(leaf.x + leaf.width, max(p1[0], p2[0], p3[0]))
            min_y = max(leaf.y, min(p1[1], p2[1], p3[1]))
            max_y = min(leaf.y + leaf.height, max(p1[1], p2[1], p3[1]))
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if self._is_point_in_triangle((x, y), p1, p2, p3): 
                        grid[y, x] = 1
    
    def _sign(self, p1, p2, p3): 
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    def _is_point_in_triangle(self, pt, v1, v2, v3):
        d1 = self._sign(pt, v1, v2)
        d2 = self._sign(pt, v2, v3)
        d3 = self._sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)


class LargeCentralHubCarver(ClearingCarver):
    """Carver for creating a large central hub with radial connections."""
    
    def carve(self, leaf, grid, config):
        # Verificar si este mapa debe tener hub central (25% de probabilidad)
        if not config.get('has_central_hub_chance', 0) or random.random() > config['has_central_hub_chance']:
            return
            
        # Solo crear el hub en una leaf que esté cerca del centro
        center_x = config['map_width'] // 2
        center_y = config['map_height'] // 2
        
        # Verificar si esta leaf está lo suficientemente cerca del centro
        leaf_center_x = leaf.x + leaf.width // 2
        leaf_center_y = leaf.y + leaf.height // 2
        
        distance_to_center = ((leaf_center_x - center_x) ** 2 + (leaf_center_y - center_y) ** 2) ** 0.5
        max_distance = min(config['map_width'], config['map_height']) * 0.15
        
        if distance_to_center > max_distance:
            return
            
        # Tamaño del hub - más moderado para dejar espacio
        hub_width = int(config['map_width'] * 0.25)
        hub_height = int(config['map_height'] * 0.25)
        
        print(f"¡CREANDO HUB CENTRAL en el centro del mapa! Tamaño: {hub_width}x{hub_height}")
        
        # Tallar el hub central
        self._draw_ellipse(center_x, center_y, hub_width, hub_height, grid, config, 1)
        
        # Crear caminos radiales que conecten el hub con los bordes
        self._create_radial_connections(grid, config, (center_y, center_x), hub_width, hub_height)
    
    def _create_radial_connections(self, grid, config, hub_center, hub_width, hub_height):
        """Crea conexiones radiales desde el hub hacia los bordes del mapa."""
        hub_y, hub_x = hub_center
        num_connections = 6
        
        for i in range(num_connections):
            angle = (2 * math.pi * i) / num_connections
            # Calcular punto en el borde del hub
            hub_edge_x = hub_x + int(hub_width * math.cos(angle))
            hub_edge_y = hub_y + int(hub_height * math.sin(angle))
            
            # Calcular punto en el borde del mapa
            if abs(math.cos(angle)) > abs(math.sin(angle)):
                if math.cos(angle) > 0:
                    map_edge_x = config['map_width'] - 1
                else:
                    map_edge_x = 0
                map_edge_y = hub_y + int((map_edge_x - hub_x) * math.tan(angle))
            else:
                if math.sin(angle) > 0:
                    map_edge_y = config['map_height'] - 1
                else:
                    map_edge_y = 0
                map_edge_x = hub_x + int((map_edge_y - hub_y) / math.tan(angle))
            
            # Asegurarse de que los puntos estén dentro del mapa
            map_edge_x = max(0, min(config['map_width'] - 1, map_edge_x))
            map_edge_y = max(0, min(config['map_height'] - 1, map_edge_y))
            
            # Crear camino de conexión
            self._create_straight_connection(grid, (hub_edge_y, hub_edge_x), 
                                            (map_edge_y, map_edge_x), config['path_width'] * 2)
    
    def _create_straight_connection(self, grid, start, end, width):
        """Crea una conexión recta entre dos puntos."""
        start_y, start_x = start
        end_y, end_x = end
        
        steps = max(abs(end_x - start_x), abs(end_y - start_y))
        if steps == 0:
            return
            
        radius = max(1, width // 2)
        
        for i in range(steps + 1):
            t = i / steps
            x = int(start_x + t * (end_x - start_x))
            y = int(start_y + t * (end_y - start_y))
            
            # Tallar el camino
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            grid[ny, nx] = 1


__all__ = [
    'ClearingCarver',
    'BaseEllipseCarver',
    'CircleCarver',
    'HorizontalEllipseCarver',
    'VerticalEllipseCarver',
    'MediumHorizontalCavernCarver',
    'RandomWalkCarver',
    'RectangleCarver',
    'TriangleCarver',
    'LargeCentralHubCarver',
]
