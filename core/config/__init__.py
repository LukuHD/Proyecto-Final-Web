"""
Environment configuration module.

Contains all environment presets and configuration classes for different map types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

# --- CONFIGURACIONES DE ENTORNOS ---
ENVIRONMENT_PRESETS = {
    'dungeon': {
        'map_width': 50, 'map_height': 30, 'min_leaf_size': 6, 'room_min_size': 4,
        'room_shape_biases': ['normal', 'wide', 'tall'],
        'generator_algorithm': 'bsp'
    },
    'forest': {
        'map_width': 71, 
        'map_height': 51,
        'generator_algorithm': 'hybrid_forest',
        'min_leaf_size': 12,
        'forest_density': 0.15,
        'path_width': 2,
        'extra_path_connections_prob': 0.25,
        'path_wiggle_room': 0.3,
        'num_entrances': 5,
        'entrance_width': 3,
        # NUEVO: PROBABILIDAD DE HUB CENTRAL (1 en 4)
        'has_central_hub_chance': 0.6,
        # NUEVO: OBSTÁCULOS PARA EL BOSQUE - usando los mismos placers
        'obstacles': [
            # Rocas en el bosque - menos cantidad y más pequeñas
            {'type': 'rock_cluster', 'count': 6},
            # Troncos en el bosque - más delgados (1 espacio de ancho)
            {'type': 'log', 'count': 8, 'min_length': 3, 'max_length': 6, 'thickness': 1}
        ],
        'clearing_carvers': [
            {'type': 'horizontal_ellipse', 'weight': 1},
            {'type': 'vertical_ellipse', 'weight': 3},
            {'type': 'circle', 'weight': 2},
            {'type': 'random_walk', 'weight': 4},
            {'type': 'rectangle', 'weight': 2},
            {'type': 'triangle', 'weight': 1},
            {'type': 'medium_horizontal_cavern', 'weight': 5},
            {'type': 'large_central_hub', 'weight': 8}
        ],
        'ellipse_params': {
            'num_shapes': (6, 10), 
            'min_radius': 2, 
            'max_radius': 5,
            'stretch_factor': 1.8,
            'medium_stretch_factor': 2.5,
        },
        'random_walk_carver': { 'steps': 150, 'brush_radius': 2 },
        'rectangle_carver': { 'num_shapes': (2, 5), 'min_width': 3, 'max_width': 8, 'min_height': 3, 'max_height': 8 },
        'triangle_carver': { 'num_shapes': (1, 3) },
        'cellular_automata_passes': 3,
        'cellular_automata_birth_limit': 4, 
        'cellular_automata_death_limit': 3
    },
    'path_focused': {
        'map_width': 80, 
        'map_height': 25,
        'generator_algorithm': 'path_focused',
        'path_width': 6,
        'path_wiggle_room': 0.8,
        'num_bifurcations': 12,
        'bifurcation_length': (8, 20),
        'bifurcation_width': 4,
        'path_styles': ['straight', 'gentle_curve', 's_curve', 'natural'],
        'path_style_weights': [2, 3, 2, 3],
        'max_curve_deviation': 0.3,
        'min_straight_segments': 3,
        'max_straight_segments': 8,
        # CONFIGURACIÓN CON MUCHOS MENOS OBSTÁCULOS
        'obstacles': [
            # Pocas rocas grandes dentro de caminos
            {'type': 'rock_cluster', 'count': 2},
            # Pocos troncos dentro de caminos
            {'type': 'log', 'count': 3, 'min_length': 4, 'max_length': 6, 'thickness': 2}
        ]
    }
}


class EnvironmentConfig(ABC):
    """Abstract base class for environment configurations."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for this environment."""
        pass


class DungeonConfig(EnvironmentConfig):
    """Configuration for dungeon-type environments."""
    
    def get_config(self) -> Dict[str, Any]:
        return ENVIRONMENT_PRESETS['dungeon']


class ForestConfig(EnvironmentConfig):
    """Configuration for forest-type environments."""
    
    def get_config(self) -> Dict[str, Any]:
        return ENVIRONMENT_PRESETS['forest']


class PathConfig(EnvironmentConfig):
    """Configuration for path-focused environments."""
    
    def get_config(self) -> Dict[str, Any]:
        return ENVIRONMENT_PRESETS['path_focused']


__all__ = [
    'ENVIRONMENT_PRESETS',
    'EnvironmentConfig',
    'DungeonConfig',
    'ForestConfig',
    'PathConfig',
]
