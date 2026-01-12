"""
Main module for procedural map generation (la.py).

This module provides backward-compatible imports and the main MapGenerator class
that orchestrates map generation using the layered architecture in the core package.

The code has been refactored into a layered structure:
- core/config/: Environment configurations and presets
- core/strategies/: Map generation strategies (BSP, Hybrid Forest, Path-focused)
- core/carvers/: Clearing carvers for creating open spaces
- core/obstacles/: Obstacle placement systems
- core/factories/: Abstract factory pattern for environment creation
"""

from copy import deepcopy
from typing import Dict, Any

# Import all components from core modules for backward compatibility
from core.config import (
    ENVIRONMENT_PRESETS,
    EnvironmentConfig,
    DungeonConfig,
    ForestConfig,
    PathConfig
)

from core.carvers import (
    ClearingCarver,
    BaseEllipseCarver,
    CircleCarver,
    HorizontalEllipseCarver,
    VerticalEllipseCarver,
    MediumHorizontalCavernCarver,
    RandomWalkCarver,
    RectangleCarver,
    TriangleCarver,
    LargeCentralHubCarver
)

from core.obstacles import (
    ObstaclePlacer,
    StrategicObstaclePlacer,
    StrategicRockClusterPlacer,
    StrategicLogPlacer
)

from core.strategies import (
    MapGenerationStrategy,
    BspDungeonStrategy,
    HybridForestStrategy,
    EnhancedPathFocusedStrategy,
    PathFocusedStrategy
)

from core.factories import (
    AbstractMapFactory,
    DungeonFactory,
    RobustForestFactory,
    RobustPathFactory
)


class MapGenerator:
    """
    Main class for generating procedural maps.
    
    Supports dynamic factory registration and multiple environment types.
    """
    
    _factories: Dict[str, AbstractMapFactory] = {}
    
    @classmethod
    def register_factory(cls, environment_type: str, factory: AbstractMapFactory):
        """Register a new factory for an environment type."""
        cls._factories[environment_type] = factory
        print(f"Fábrica registrada para: {environment_type}")

    @classmethod
    def get_registered_factories(cls):
        """Get all registered factories."""
        return cls._factories.copy()

    @classmethod
    def get_factory_info(cls, environment_type: str):
        """Get information about a specific factory."""
        factory = cls._factories.get(environment_type)
        if factory:
            return factory.get_environment_info()
        return None

    # Static factory map for algorithms
    FACTORY_MAP = {
        'bsp': DungeonFactory(),
        'hybrid_forest': RobustForestFactory(),
        'path_focused': RobustPathFactory(),
    }

    def __init__(self, environment_type='dungeon'):
        """
        Initialize the map generator.
        
        Args:
            environment_type: Type of environment to generate ('dungeon', 'forest', 'path_focused')
        """
        if environment_type not in ENVIRONMENT_PRESETS:
            raise ValueError(f"Tipo de entorno no válido: '{environment_type}'")

        self.config = deepcopy(ENVIRONMENT_PRESETS[environment_type])
        algorithm_name = self.config.get('generator_algorithm')

        factory = self._factories.get(environment_type) or self.FACTORY_MAP.get(algorithm_name)

        if factory is None:
            raise ValueError(f"No hay fábrica para: {environment_type} (algoritmo: {algorithm_name})")

        print(f"Usando fábrica: {factory.__class__.__name__}")
        self.strategy = factory.create_strategy()
        self.factory = factory

    def generate(self):
        """Generate a map using the current strategy and configuration."""
        return self.strategy.generate(self.config)
    
    def get_strategy_info(self):
        """Get information about the current strategy."""
        return {
            "strategy_name": self.strategy.get_name(),
            "supported_features": self.strategy.get_supported_features(),
            "factory_info": self.factory.get_environment_info()
        }


def print_map(grid, environment_type):
    """
    Print a visual representation of the generated map.
    
    Args:
        grid: The numpy array representing the map
        environment_type: The type of environment for appropriate symbols
    """
    if 'path_focused' in environment_type:
        chars = {
            0: ' ',    # Espacio vacío
            1: '·',    # Camino
            2: '■',    # Rocas (bloques sólidos)
            3: '≡'     # Troncos
        }
    elif 'forest' in environment_type:
        chars = {
            0: '♣',    # Árboles/bosque
            1: '.',    # Camino
            2: '■',    # Rocas
            3: '≡'     # Troncos
        }
    else:  # dungeon
        chars = {
            0: '#',    # Pared
            1: '.',    # Suelo
            2: '■',    # Rocas
            3: '≡'     # Troncos
        }
        
    print(f"--- Mapa generado: {environment_type.replace('_', ' ').title()} ---")
    for row in grid:
        print("".join([chars.get(cell, '?') for cell in row]))
    print("\n")


# Main execution block
if __name__ == "__main__":
    # Registrar fábricas dinámicamente
    MapGenerator.register_factory('dungeon', DungeonFactory())
    MapGenerator.register_factory('forest', RobustForestFactory())
    MapGenerator.register_factory('path_focused', RobustPathFactory())
    
    print("=== SISTEMA DE GENERACIÓN DE MAPAS MEJORADO ===\n")
    
    # Probar cada tipo de entorno
    environments = ['dungeon', 'forest', 'path_focused']
    
    for env in environments:
        try:
            print(f"Generando mapa de tipo: {env}")
            generator = MapGenerator(environment_type=env)
            
            # Mostrar información de la estrategia
            info = generator.get_strategy_info()
            print(f"Estrategia: {info['strategy_name']}")
            print(f"Características: {', '.join(info['supported_features'])}")
            print(f"Descripción: {info['factory_info']['description']}")
            
            # Generar y mostrar el mapa
            map_grid = generator.generate()
            print_map(map_grid, env)
            
        except Exception as e:
            print(f"Error generando {env}: {e}")
            print()
