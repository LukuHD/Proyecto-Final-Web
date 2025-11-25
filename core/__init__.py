"""
Core module for procedural map generation.

This module provides a layered architecture for generating procedural maps
including dungeons, forests, and path-focused environments.

Layers:
- config: Environment configurations and presets
- strategies: Map generation strategies (BSP, Hybrid Forest, Path-focused)
- carvers: Clearing carvers for creating open spaces
- obstacles: Obstacle placement systems
- factories: Abstract factory pattern for environment creation
"""

from core.config import ENVIRONMENT_PRESETS, EnvironmentConfig, DungeonConfig, ForestConfig, PathConfig
from core.factories import (
    AbstractMapFactory,
    DungeonFactory,
    RobustForestFactory,
    RobustPathFactory
)
from core.strategies import (
    MapGenerationStrategy,
    BspDungeonStrategy,
    HybridForestStrategy,
    EnhancedPathFocusedStrategy,
    PathFocusedStrategy
)
from core.carvers import (
    ClearingCarver,
    BaseEllipseCarver,
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
    StrategicObstaclePlacer,
    StrategicRockClusterPlacer,
    StrategicLogPlacer
)

__all__ = [
    # Config
    'ENVIRONMENT_PRESETS',
    'EnvironmentConfig',
    'DungeonConfig',
    'ForestConfig',
    'PathConfig',
    # Factories
    'AbstractMapFactory',
    'DungeonFactory',
    'RobustForestFactory',
    'RobustPathFactory',
    # Strategies
    'MapGenerationStrategy',
    'BspDungeonStrategy',
    'HybridForestStrategy',
    'EnhancedPathFocusedStrategy',
    'PathFocusedStrategy',
    # Carvers
    'ClearingCarver',
    'BaseEllipseCarver',
    'CircleCarver',
    'HorizontalEllipseCarver',
    'VerticalEllipseCarver',
    'RandomWalkCarver',
    'RectangleCarver',
    'TriangleCarver',
    'MediumHorizontalCavernCarver',
    'LargeCentralHubCarver',
    # Obstacles
    'ObstaclePlacer',
    'StrategicObstaclePlacer',
    'StrategicRockClusterPlacer',
    'StrategicLogPlacer',
]
