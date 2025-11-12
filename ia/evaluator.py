"""
MapEvaluator: Evalúa mapas usando métricas ponderadas
- Densidad de habitaciones/caminos
- Densidad de obstáculos
- Tamaño promedio de habitaciones
- Puntuación basada en pesos configurables
"""

import json
import os
import numpy as np
from pathlib import Path


class MapEvaluator:
    """Evalúa mapas usando métricas ponderadas cargadas desde configuración"""
    
    def __init__(self, config_path=None):
        """
        Inicializa el evaluador con pesos desde archivo de configuración
        
        Args:
            config_path: Ruta al archivo weights.json. Si es None, usa la ruta por defecto.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "weights.json"
        
        self.config_path = config_path
        self.weights = self._load_weights()
    
    def _load_weights(self):
        """Carga los pesos desde el archivo de configuración"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return data.get('weights', self._get_default_weights())
        except FileNotFoundError:
            print(f"Advertencia: No se encontró {self.config_path}, usando pesos por defecto")
            return self._get_default_weights()
        except json.JSONDecodeError:
            print(f"Advertencia: Error al leer {self.config_path}, usando pesos por defecto")
            return self._get_default_weights()
    
    def _get_default_weights(self):
        """Retorna pesos balanceados por defecto"""
        return {
            'room_density': 0.25,
            'path_density': 0.25,
            'obstacle_density': 0.15,
            'avg_room_size': 0.20,
            'connectivity': 0.15
        }
    
    def calculate_room_density(self, map_grid):
        """
        Calcula la densidad de habitaciones/caminos (celdas transitables)
        
        Args:
            map_grid: numpy array del mapa (0=pared/vacío, 1=transitable, 2+=obstáculos)
        
        Returns:
            float: Densidad de habitaciones (0.0 a 1.0)
        """
        total_cells = map_grid.size
        floor_cells = np.count_nonzero(map_grid == 1)
        return floor_cells / total_cells if total_cells > 0 else 0.0
    
    def calculate_path_density(self, map_grid):
        """
        Calcula la densidad de caminos basándose en celdas con exactamente 2 vecinos
        (indicativo de pasillos)
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            float: Densidad de caminos (0.0 a 1.0)
        """
        height, width = map_grid.shape
        path_cells = 0
        floor_cells = np.count_nonzero(map_grid == 1)
        
        if floor_cells == 0:
            return 0.0
        
        for y in range(height):
            for x in range(width):
                if map_grid[y, x] == 1:
                    # Contar vecinos transitables (4-conectividad)
                    neighbors = 0
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and map_grid[ny, nx] == 1:
                            neighbors += 1
                    
                    # Si tiene exactamente 2 vecinos opuestos, es un pasillo
                    if neighbors == 2:
                        path_cells += 1
        
        return path_cells / floor_cells if floor_cells > 0 else 0.0
    
    def calculate_obstacle_density(self, map_grid):
        """
        Calcula la densidad de obstáculos (valores >= 2)
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            float: Densidad de obstáculos (0.0 a 1.0)
        """
        total_cells = map_grid.size
        obstacle_cells = np.count_nonzero(map_grid >= 2)
        return obstacle_cells / total_cells if total_cells > 0 else 0.0
    
    def calculate_avg_room_size(self, map_grid):
        """
        Calcula el tamaño promedio de habitaciones usando componentes conexas
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            float: Tamaño promedio normalizado de habitaciones (0.0 a 1.0)
        """
        from collections import deque
        
        height, width = map_grid.shape
        visited = np.zeros_like(map_grid, dtype=bool)
        room_sizes = []
        
        def bfs(start_y, start_x):
            """Breadth-first search para encontrar componente conexa"""
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            size = 0
            
            while queue:
                y, x = queue.popleft()
                size += 1
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and 
                        map_grid[ny, nx] == 1 and not visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            
            return size
        
        # Encontrar todas las componentes conexas (habitaciones)
        for y in range(height):
            for x in range(width):
                if map_grid[y, x] == 1 and not visited[y, x]:
                    room_size = bfs(y, x)
                    room_sizes.append(room_size)
        
        if not room_sizes:
            return 0.0
        
        avg_size = np.mean(room_sizes)
        # Normalizar: asumiendo que un mapa típico tiene habitaciones de ~50-200 celdas
        # Normalizar a un rango razonable
        max_expected_room = 200
        normalized = min(avg_size / max_expected_room, 1.0)
        
        return normalized
    
    def calculate_connectivity(self, map_grid):
        """
        Calcula qué tan conectado está el mapa (qué % de celdas transitables son alcanzables)
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            float: Conectividad (0.0 a 1.0)
        """
        from collections import deque
        
        height, width = map_grid.shape
        floor_cells = np.argwhere(map_grid == 1)
        
        if len(floor_cells) == 0:
            return 0.0
        
        # Empezar BFS desde la primera celda transitable
        start = tuple(floor_cells[0])
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            y, x = queue.popleft()
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and 
                    map_grid[ny, nx] == 1 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        return len(visited) / len(floor_cells) if len(floor_cells) > 0 else 0.0
    
    def calculate_metrics(self, map_grid):
        """
        Calcula todas las métricas para un mapa
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        return {
            'room_density': self.calculate_room_density(map_grid),
            'path_density': self.calculate_path_density(map_grid),
            'obstacle_density': self.calculate_obstacle_density(map_grid),
            'avg_room_size': self.calculate_avg_room_size(map_grid),
            'connectivity': self.calculate_connectivity(map_grid)
        }
    
    def score(self, map_grid):
        """
        Calcula la puntuación total del mapa basada en pesos configurables
        
        Args:
            map_grid: numpy array del mapa
        
        Returns:
            tuple: (score, metrics_dict) - puntuación total y métricas individuales
        """
        metrics = self.calculate_metrics(map_grid)
        
        # Calcular puntuación ponderada
        score = sum(metrics[key] * self.weights[key] for key in metrics.keys())
        
        return score, metrics
    
    def reload_weights(self):
        """Recarga los pesos desde el archivo de configuración"""
        self.weights = self._load_weights()
