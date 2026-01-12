"""
Obstacle placement module.

Contains all obstacle placer classes for adding obstacles to maps.
"""

import random
from abc import ABC, abstractmethod


class ObstaclePlacer(ABC):
    """Abstract base class for obstacle generators."""
    
    @abstractmethod
    def place(self, grid, path_coords, config):
        """Place obstacles on the grid."""
        pass


class StrategicObstaclePlacer(ABC):
    """Base class for obstacle generators with strategic placement logic."""
    
    @abstractmethod
    def place(self, grid, path_coords, config):
        """Place obstacles on the grid."""
        pass
    
    def _is_valid_obstacle_position(self, grid, x, y, min_distance_from_path=1):
        """Verifica si una posición es válida para colocar un obstáculo."""
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return False
        
        # Permitir obstáculos cerca de caminos para obstruirlos
        # Pero asegurarse de que no bloqueen completamente
        for dy in range(-min_distance_from_path, min_distance_from_path + 1):
            for dx in range(-min_distance_from_path, min_distance_from_path + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    if grid[ny, nx] == 1:  # Es un camino
                        # Verificar que no bloquee completamente
                        if not self._allows_path_passage(grid, x, y, min_distance_from_path):
                            return False
        return True
    
    def _allows_path_passage(self, grid, x, y, radius):
        """Verifica que colocar un obstáculo aquí no bloquee completamente el paso."""
        # Contar caminos adyacentes
        adjacent_paths = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    if grid[ny, nx] == 1:
                        adjacent_paths += 1
        
        # Si hay suficientes caminos adyacentes, permite el paso
        return adjacent_paths >= 2


class StrategicRockClusterPlacer(StrategicObstaclePlacer):
    """Coloca grupos de rocas de tamaño 2x2 o más dentro de los caminos."""
    
    def place(self, grid, path_coords, config):
        map_height, map_width = grid.shape
        count = config.get('count', 1)
        
        # Tamaños de rocas: mínimo 2x2, máximo 4x4
        rock_sizes = [
            (2, 2), (2, 2), (2, 2),  # Principalmente 2x2
            (3, 2), (2, 3),           # Algunas ovaladas
            (3, 3)                    # Pocas más grandes
        ]
        
        print(f"Colocando {count} grupos de rocas (2x2 mínimo) dentro de caminos")
        
        placed = 0
        attempts = 0
        max_attempts = count * 20  # Más intentos para encontrar buenas posiciones
        
        # Filtrar posiciones para evitar el camino principal
        side_path_coords = self._get_side_path_coords(grid, path_coords)
        
        while placed < count and attempts < max_attempts:
            attempts += 1
            
            # Preferir caminos laterales sobre el principal
            if side_path_coords and random.random() < 0.7:  # 70% en caminos laterales
                center_y, center_x = random.choice(side_path_coords)
            elif path_coords:
                center_y, center_x = random.choice(path_coords)
            else:
                break
            
            # Elegir un tamaño de roca (mínimo 2x2)
            rx, ry = random.choice(rock_sizes)
            
            # Verificar que podemos colocar la roca aquí
            can_place = True
            rocks_to_place = []
            
            for x_offset in range(-rx, rx + 1):
                for y_offset in range(-ry, ry + 1):
                    # Fórmula de elipse para formas más naturales
                    if (x_offset**2 / max(1, rx**2)) + (y_offset**2 / max(1, ry**2)) <= 1:
                        x, y = center_x + x_offset, center_y + y_offset
                        if (0 <= y < map_height and 0 <= x < map_width):
                            # Solo colocar en caminos existentes y que no tengan obstáculos
                            if grid[y, x] == 1:  # Es un camino sin obstáculos
                                rocks_to_place.append((y, x))
                            else:
                                can_place = False
                                break
                if not can_place:
                    break
            
            # Verificar que tenemos al menos 4 celdas (2x2 mínimo)
            if can_place and len(rocks_to_place) >= 4:
                # Colocar las rocas
                for y, x in rocks_to_place:
                    grid[y, x] = 2  # 2 para rocas
                placed += 1
                print(f"  - Roca {placed}: {rx}x{ry} en ({center_x}, {center_y}) con {len(rocks_to_place)} celdas")
    
    def _get_side_path_coords(self, grid, path_coords):
        """Obtiene coordenadas de caminos que no son el principal."""
        if not path_coords:
            return []
        
        # Identificar el camino principal (asumiendo que es el más largo)
        # En una implementación más sofisticada, podríamos etiquetar caminos
        # Por ahora, simplemente tomamos una muestra aleatoria
        return random.sample(path_coords, min(len(path_coords) // 2, len(path_coords)))


class StrategicLogPlacer(StrategicObstaclePlacer):
    """Coloca troncos dentro de caminos de manera estratégica."""
    
    def place(self, grid, path_coords, config):
        map_height, map_width = grid.shape
        count = config.get('count', 1)
        min_len = config.get('min_length', 3)
        max_len = config.get('max_length', 6)
        thickness = 2  # 2 espacios de ancho
        
        print(f"Colocando {count} troncos dentro de caminos")
        
        placed = 0
        attempts = 0
        max_attempts = count * 25
        
        # Filtrar posiciones para evitar el camino principal
        side_path_coords = self._get_side_path_coords(grid, path_coords)
        
        while placed < count and attempts < max_attempts:
            attempts += 1
            
            # Preferir caminos laterales sobre el principal
            if side_path_coords and random.random() < 0.7:  # 70% en caminos laterales
                start_y, start_x = random.choice(side_path_coords)
            elif path_coords:
                start_y, start_x = random.choice(path_coords)
            else:
                break
            
            # Orientación basada en la forma del camino
            horizontal = self._should_be_horizontal(grid, start_x, start_y)
            length = random.randint(min_len, max_len)
            
            # Ajustar posición para que el tronco quede mejor alineado
            start_x, start_y = self._adjust_start_position(grid, start_x, start_y, length, horizontal, thickness)
            
            # Verificar que el tronco quepa y esté dentro del camino
            if self._is_valid_log_position(grid, start_x, start_y, length, horizontal, thickness):
                self._place_log(grid, start_x, start_y, length, horizontal, thickness)
                placed += 1
                print(f"  - Tronco {placed}: {length}x{thickness} en ({start_x}, {start_y}), {'horizontal' if horizontal else 'vertical'}")
    
    def _get_side_path_coords(self, grid, path_coords):
        """Obtiene coordenadas de caminos que no son el principal."""
        if not path_coords:
            return []
        return random.sample(path_coords, min(len(path_coords) // 2, len(path_coords)))
    
    def _should_be_horizontal(self, grid, x, y):
        """Determina si el tronco debe ser horizontal basado en el entorno."""
        horizontal_path = 0
        vertical_path = 0
        
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx] == 1:
                    if abs(dx) > abs(dy):
                        horizontal_path += 1
                    else:
                        vertical_path += 1
        
        return horizontal_path < vertical_path
    
    def _adjust_start_position(self, grid, start_x, start_y, length, horizontal, thickness):
        """Ajusta la posición inicial para que el tronco quede mejor alineado con el camino."""
        if horizontal:
            for offset in range(-2, 3):
                new_y = start_y + offset
                if 0 <= new_y < grid.shape[0] and self._has_horizontal_clearance(grid, start_x, new_y, length, thickness):
                    return start_x, new_y
        else:
            for offset in range(-2, 3):
                new_x = start_x + offset
                if 0 <= new_x < grid.shape[1] and self._has_vertical_clearance(grid, new_x, start_y, length, thickness):
                    return new_x, start_y
        
        return start_x, start_y
    
    def _has_horizontal_clearance(self, grid, x, y, length, thickness):
        """Verifica si hay espacio horizontal continuo para un tronco."""
        if x + length >= grid.shape[1] or y + thickness >= grid.shape[0]:
            return False
            
        for i in range(length):
            for t in range(thickness):
                if grid[y + t, x + i] != 1:
                    return False
        return True
    
    def _has_vertical_clearance(self, grid, x, y, length, thickness):
        """Verifica si hay espacio vertical continuo para un tronco."""
        if y + length >= grid.shape[0] or x + thickness >= grid.shape[1]:
            return False
            
        for i in range(length):
            for t in range(thickness):
                if grid[y + i, x + t] != 1:
                    return False
        return True
    
    def _is_valid_log_position(self, grid, start_x, start_y, length, horizontal, thickness):
        """Verifica si es seguro colocar un tronco en esta posición."""
        if horizontal:
            if start_x + length >= grid.shape[1] or start_y + thickness >= grid.shape[0]:
                return False
            for i in range(length):
                for t in range(thickness):
                    x, y = start_x + i, start_y + t
                    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
                        return False
                    if grid[y, x] != 1:
                        return False
        else:
            if start_y + length >= grid.shape[0] or start_x + thickness >= grid.shape[1]:
                return False
            for i in range(length):
                for t in range(thickness):
                    x, y = start_x + t, start_y + i
                    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
                        return False
                    if grid[y, x] != 1:
                        return False
        
        return True
    
    def _place_log(self, grid, start_x, start_y, length, horizontal, thickness):
        """Coloca un tronco en la posición especificada."""
        for i in range(length):
            for t in range(thickness):
                if horizontal:
                    x, y = start_x + i, start_y + t
                else:
                    x, y = start_x + t, start_y + i
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == 1:
                    grid[y, x] = 3  # 3 para troncos


__all__ = [
    'ObstaclePlacer',
    'StrategicObstaclePlacer',
    'StrategicRockClusterPlacer',
    'StrategicLogPlacer',
]
