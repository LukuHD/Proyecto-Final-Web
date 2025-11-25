"""
Factory module for map generation.

Contains all factory classes for creating map generation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from core.config import DungeonConfig, ForestConfig, PathConfig
from core.strategies import (
    MapGenerationStrategy,
    BspDungeonStrategy,
    HybridForestStrategy,
    EnhancedPathFocusedStrategy
)
from core.carvers import (
    ClearingCarver,
    CircleCarver,
    HorizontalEllipseCarver,
    VerticalEllipseCarver,
    RandomWalkCarver,
    RectangleCarver,
    TriangleCarver,
    MediumHorizontalCavernCarver,
    LargeCentralHubCarver
)
from core.obstacles import (
    ObstaclePlacer,
    StrategicRockClusterPlacer,
    StrategicLogPlacer
)


class AbstractMapFactory(ABC):
    """
    Abstract Factory interface for creating map generation strategies.
    """
    
    @abstractmethod
    def create_strategy(self) -> MapGenerationStrategy:
        """Crea la estrategia principal de generación."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validar configuración antes de crear la estrategia."""
        required = ['map_width', 'map_height', 'generator_algorithm']
        return all(key in config for key in required)
    
    def create_map(self, config: Dict[str, Any]):
        """Método template para creación completa del mapa."""
        strategy = self.create_strategy()
        return strategy.generate(config)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Información sobre el entorno que genera esta fábrica."""
        return {
            "type": self.__class__.__name__.replace("Factory", "").lower(),
            "description": "Generador de mapas personalizado"
        }


class DungeonFactory(AbstractMapFactory):
    """Concrete factory for dungeon-type maps."""
    
    def create_strategy(self) -> MapGenerationStrategy:
        config = DungeonConfig().get_config()
        if not self.validate_config(config):
            raise ValueError("Configuración de mazmorra incompleta")
        return BspDungeonStrategy()
    
    def get_environment_info(self) -> Dict[str, Any]:
        base_info = super().get_environment_info()
        base_info.update({
            "description": "Generador de mazmorras con habitaciones y pasillos usando BSP",
            "features": ["rooms", "corridors", "bsp_tree"]
        })
        return base_info


class RobustForestFactory(AbstractMapFactory):
    """Concrete factory for forest-type maps with robust error handling."""
    
    def create_strategy(self) -> MapGenerationStrategy:
        try:
            config = ForestConfig().get_config()
            if not self.validate_config(config):
                raise ValueError("Configuración de bosque incompleta")
            
            carver_map = self._create_carvers()
            obstacle_placers = self._create_obstacle_placers()
            
            if not carver_map:
                raise ValueError("No se pudieron crear los carvers")
                
            return HybridForestStrategy(carver_map=carver_map, obstacle_placers=obstacle_placers)
            
        except KeyError:
            raise ValueError("Configuración de bosque no encontrada")
        except Exception as e:
            raise RuntimeError(f"Error creando estrategia de bosque: {e}")
    
    def _create_carvers(self) -> Dict[str, ClearingCarver]:
        """Crear todos los carvers con validación."""
        carvers = {}
        carver_classes = {
            'circle': CircleCarver,
            'horizontal_ellipse': HorizontalEllipseCarver,
            'vertical_ellipse': VerticalEllipseCarver,
            'random_walk': RandomWalkCarver,
            'rectangle': RectangleCarver,
            'triangle': TriangleCarver,
            'medium_horizontal_cavern': MediumHorizontalCavernCarver,
            'large_central_hub': LargeCentralHubCarver
        }
        
        for name, carver_class in carver_classes.items():
            try:
                carvers[name] = carver_class()
            except Exception as e:
                print(f"Advertencia: No se pudo crear {name}: {e}")
                
        return carvers
    
    def _create_obstacle_placers(self) -> Dict[str, ObstaclePlacer]:
        """Crear obstacle placers para el bosque."""
        placers = {}
        placer_classes = {
            'rock_cluster': StrategicRockClusterPlacer,
            'log': StrategicLogPlacer
        }
        
        for name, placer_class in placer_classes.items():
            try:
                placers[name] = placer_class()
            except Exception as e:
                print(f"Advertencia: No se pudo crear {name}: {e}")
                
        return placers
    
    def get_environment_info(self) -> Dict[str, Any]:
        base_info = super().get_environment_info()
        base_info.update({
            "description": "Generador de bosques con claros, caminos sinuosos y obstáculos naturales",
            "features": ["clearings", "paths", "cellular_automata", "entrances", "natural_obstacles"]
        })
        return base_info


class RobustPathFactory(AbstractMapFactory):
    """Concrete factory for path-focused maps with robust error handling."""
    
    def create_strategy(self) -> MapGenerationStrategy:
        try:
            config = PathConfig().get_config()
            if not self.validate_config(config):
                raise ValueError("Configuración de camino incompleta")
            
            placer_map = self._create_obstacle_placers()
            
            if not placer_map:
                raise ValueError("No se pudieron crear los obstacle placers")
                
            return EnhancedPathFocusedStrategy(obstacle_placers=placer_map)
            
        except KeyError:
            raise ValueError("Configuración de camino no encontrada")
        except Exception as e:
            raise RuntimeError(f"Error creando estrategia de camino: {e}")
    
    def _create_obstacle_placers(self) -> Dict[str, ObstaclePlacer]:
        """Crear todos los obstacle placers con validación."""
        placers = {}
        placer_classes = {
            'rock_cluster': StrategicRockClusterPlacer,
            'log': StrategicLogPlacer
        }
        
        for name, placer_class in placer_classes.items():
            try:
                placers[name] = placer_class()
            except Exception as e:
                print(f"Advertencia: No se pudo crear {name}: {e}")
                
        return placers
    
    def get_environment_info(self) -> Dict[str, Any]:
        base_info = super().get_environment_info()
        base_info.update({
            "description": "Generador de caminos con obstáculos estratégicos y bifurcaciones",
            "features": ["main_path", "bifurcations", "strategic_obstacles", "natural_paths"]
        })
        return base_info


__all__ = [
    'AbstractMapFactory',
    'DungeonFactory',
    'RobustForestFactory',
    'RobustPathFactory',
]
