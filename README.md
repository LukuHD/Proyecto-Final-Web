# Proyecto-Final-Web
Diseñar e implementar mapas de manera procedural regulados por IA.

## Estructura del Proyecto

```
├── main.py                 # API FastAPI para generación de mapas
├── la.py                   # Módulo principal de generación de mapas
├── generator.py            # Utilidades de generación de mazmorras
├── core/                   # Paquete con arquitectura en capas
│   ├── config/             # Configuraciones de entornos (dungeon, forest, path_focused)
│   ├── strategies/         # Estrategias de generación (BSP, Hybrid Forest, Path-focused)
│   ├── carvers/            # Talladores de claros (círculos, elipses, etc.)
│   ├── obstacles/          # Colocadores de obstáculos (rocas, troncos)
│   └── factories/          # Fábricas abstractas para creación de estrategias
├── quality_agent/          # Agente de calidad con aprendizaje por refuerzo
│   ├── learning/           # Modelos de preferencia y cálculo de recompensas
│   ├── storage/            # Almacenamiento de preferencias
│   └── metrics.py          # Métricas de calidad de mapas
├── test_learning_agent.py  # Tests del agente de aprendizaje
└── train_maps_with_ai.py   # Entrenamiento de modelos con IA
```

## Tipos de Mapas Soportados

- **Dungeon**: Mazmorras generadas con BSP (Binary Space Partition)
- **Forest**: Bosques con claros, caminos y obstáculos naturales
- **Path Focused**: Mapas centrados en caminos con bifurcaciones

## Uso

```python
from la import MapGenerator, print_map

# Generar un mapa de mazmorra
generator = MapGenerator('dungeon')
grid = generator.generate()
print_map(grid, 'dungeon')

# Generar un mapa de bosque
generator = MapGenerator('forest')
grid = generator.generate()
print_map(grid, 'forest')
```
