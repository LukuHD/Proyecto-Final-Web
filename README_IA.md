# Sistema de IA con Aprendizaje por Refuerzo para Mapas Procedurales

Este proyecto implementa un sistema de Inteligencia Artificial que aprende de las preferencias del usuario para generar mapas procedurales personalizados usando aprendizaje por refuerzo.

## 🎯 Características

- **Evaluación Inteligente de Mapas**: Sistema que evalúa mapas usando múltiples métricas ponderadas
- **Aprendizaje por Refuerzo**: Adapta pesos y parámetros basándose en las elecciones del usuario
- **Generación Procedural**: Genera mapas de dungeon, bosque y caminos con características únicas
- **Personalización Incremental**: El sistema mejora con cada elección del usuario

## 📁 Estructura del Proyecto

```
.
├── ia/                           # Módulo de IA
│   ├── __init__.py
│   ├── evaluator.py              # Evalúa mapas con métricas ponderadas
│   ├── adapter.py                # Aprende y ajusta pesos/parámetros
│   └── configs/
│       ├── weights.json          # Pesos actuales del evaluador
│       └── environment_adjustments.json  # Ajustes de generación
├── la.py                         # Generador de mapas procedurales
├── test.py                       # Script principal interactivo
├── demo_ia.py                    # Demo automatizada del sistema
├── test_ia_system.py             # Suite de pruebas automatizadas
└── README_IA.md                  # Esta documentación
```

## 🚀 Instalación

### Requisitos

- Python 3.8 o superior
- numpy

### Instalación de dependencias

```bash
pip install numpy
```

## 📖 Uso

### 1. Demo Rápida (Automática)

Para ver una demostración rápida del sistema sin interacción:

```bash
python3 demo_ia.py
```

Este script:
- Genera 5 mapas en cada ronda
- Selecciona automáticamente los 2 mejores
- Simula la elección del usuario (siempre el de mayor score)
- Muestra cómo el sistema aprende y ajusta pesos
- Ejecuta 3 rondas de aprendizaje

### 2. Modo Interactivo

Para usar el sistema de forma interactiva y hacer tus propias elecciones:

```bash
python3 test.py
```

El script te presentará un menú con opciones:
1. Entrenar con mapas de Dungeon
2. Entrenar con mapas de Forest
3. Entrenar con mapas de Path-Focused
4. Entrenar con tipo mixto (aleatorio)
5. Ver pesos actuales
6. Resetear pesos y ajustes
Q. Salir

En cada sesión de entrenamiento:
- Se generan 10 mapas del tipo seleccionado
- El evaluador preselecciona los 2 mejores
- Se te muestran ambos mapas con sus métricas
- Eliges tu favorito (A o B)
- El sistema aprende de tu elección y ajusta los pesos

### 3. Pruebas Automatizadas

Para ejecutar la suite completa de pruebas:

```bash
python3 test_ia_system.py
```

Esto ejecutará 5 pruebas que verifican:
- ✅ Funcionamiento del evaluador
- ✅ Funcionamiento del adaptador
- ✅ Generación y evaluación de mapas
- ✅ Ajustes de entorno
- ✅ Flujo completo del sistema

## 🧠 Cómo Funciona

### 1. Evaluador de Mapas (`ia/evaluator.py`)

El evaluador calcula 5 métricas principales para cada mapa:

- **`room_density`**: Densidad de celdas transitables (0.0 - 1.0)
- **`path_density`**: Densidad de pasillos (celdas con 2 vecinos opuestos)
- **`obstacle_density`**: Densidad de obstáculos (rocas, troncos)
- **`avg_room_size`**: Tamaño promedio normalizado de habitaciones
- **`connectivity`**: Qué tan conectado está el mapa (0.0 - 1.0)

Cada métrica tiene un **peso** asociado que determina su importancia en el score final:

```python
score = Σ (métrica[i] × peso[i])
```

### 2. Adaptador (`ia/adapter.py`)

El adaptador implementa el aprendizaje por refuerzo:

#### Ajuste de Pesos
- Incrementa los pesos de las métricas altas en el mapa ganador
- Reduce ligeramente los pesos de las métricas dominantes del mapa perdedor
- Normaliza los pesos para que sumen 1.0
- Aplica límites (min: 0.05, max: 0.50)

#### Ajuste de Parámetros de Entorno
Basándose en las métricas del ganador, ajusta:
- `min_leaf_size`: Tamaño mínimo de subdivisiones BSP (afecta densidad)
- `room_min_size`: Tamaño mínimo de habitaciones
- `path_width`: Ancho de caminos
- `obstacle_multiplier`: Multiplicador de cantidad de obstáculos
- `extra_path_connections_prob`: Probabilidad de conexiones extra

### 3. Flujo de Aprendizaje

```
┌─────────────────────┐
│  Generar N mapas    │
│  (usando la.py)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Evaluar todos      │
│  (evaluator.score)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Seleccionar top 2  │
│  (mayor score)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Mostrar al usuario │
│  Mapa A vs Mapa B   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Usuario elige      │
│  (A o B)            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Aprender           │
│  adapter.learn()    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Ajustar entorno    │
│  adjust_env_params  │
└──────────┬──────────┘
           │
           ▼
     ┌────────────┐
     │  Repetir   │
     └────────────┘
```

## 📊 Ejemplo de Evolución de Pesos

### Iteración 0 (Inicial)
```
room_density      : 0.2500
path_density      : 0.2500
obstacle_density  : 0.1500
avg_room_size     : 0.2000
connectivity      : 0.1500
```

### Iteración 1 (Después de elegir mapa con alta conectividad)
```
room_density      : 0.2357 ↓
path_density      : 0.2262 ↓
obstacle_density  : 0.1293 ↓
avg_room_size     : 0.2033 ↑
connectivity      : 0.2054 ↑
```

### Iteración 3 (Después de 3 elecciones)
```
room_density      : 0.2225 ↓
path_density      : 0.2119 ↓
obstacle_density  : 0.1147 ↓
avg_room_size     : 0.2245 ↑
connectivity      : 0.2264 ↑
```

Como se puede ver, el sistema ha aprendido que el usuario prefiere mapas con:
- ✅ Mayor conectividad
- ✅ Habitaciones más grandes
- ❌ Menos obstáculos
- ❌ Menos pasillos largos

## 🔧 Configuración

### Pesos Iniciales (`ia/configs/weights.json`)

```json
{
  "weights": {
    "room_density": 0.25,
    "path_density": 0.25,
    "obstacle_density": 0.15,
    "avg_room_size": 0.20,
    "connectivity": 0.15
  },
  "iteration": 0,
  "learning_rate": 0.1
}
```

- **`learning_rate`**: Controla qué tan rápido aprende el sistema (0.0 - 1.0)
  - Valores bajos (0.05): Aprendizaje lento pero estable
  - Valores altos (0.3): Aprendizaje rápido pero puede ser inestable

### Ajustes de Entorno (`ia/configs/environment_adjustments.json`)

Se actualiza automáticamente con cada aprendizaje. Ejemplo:

```json
{
  "dungeon": {
    "min_leaf_size": 5,
    "obstacle_multiplier": 1.1,
    "room_min_size": 6
  },
  "forest": {
    "obstacle_multiplier": 1.0,
    "path_width": 3,
    "extra_path_connections_prob": 0.3
  }
}
```

## 📝 API Reference

### MapEvaluator

```python
from ia.evaluator import MapEvaluator

evaluator = MapEvaluator()

# Evaluar un mapa
score, metrics = evaluator.score(map_grid)

# Recargar pesos desde archivo
evaluator.reload_weights()
```

### MapAdapter

```python
from ia.adapter import MapAdapter

adapter = MapAdapter(learning_rate=0.1)

# Aprender de una elección
new_weights = adapter.learn(
    winning_map_metrics,
    losing_map_metrics  # opcional
)

# Ajustar parámetros de entorno
adjustments = adapter.adjust_environment_params(
    'dungeon',  # tipo de entorno
    winning_map_metrics
)

# Obtener configuración ajustada
adjusted_config = adapter.get_adjusted_config(
    base_config,
    'dungeon'
)
```

## 🎮 Tipos de Mapas

### Dungeon (Mazmorra)
- Habitaciones rectangulares conectadas por pasillos
- Generado usando Binary Space Partitioning (BSP)
- Ideal para juegos de exploración

### Forest (Bosque)
- Claros orgánicos conectados por caminos sinuosos
- Usa autómatas celulares para formas naturales
- Obstáculos: rocas y troncos

### Path-Focused (Centrado en Caminos)
- Camino principal con bifurcaciones
- Varios estilos: recto, curvo, S-curve, natural
- Obstáculos estratégicos en el camino

## 🐛 Solución de Problemas

### El sistema no aprende correctamente
- Verifica que `learning_rate` no sea demasiado bajo (<0.05)
- Asegúrate de que los mapas tengan suficiente variación
- Reinicia el sistema con la opción 6 en el menú interactivo

### Error al generar mapas
- Verifica que numpy esté instalado: `pip install numpy`
- Algunos tipos de mapas pueden fallar ocasionalmente; el sistema continúa con los que se generan correctamente

### Los pesos no se guardan
- Verifica permisos de escritura en `ia/configs/`
- Asegúrate de que la carpeta existe

## 📄 Licencia

Este proyecto es parte del Proyecto Final Web y está diseñado con fines educativos.

## 👥 Autores

- Sistema de Generación Procedural: Equipo del Proyecto
- Sistema de IA y Aprendizaje por Refuerzo: Implementado según issue #1

## 🙏 Agradecimientos

- Binary Space Partitioning (BSP) para generación de dungeons
- Autómatas celulares para formas orgánicas
- Algoritmo A* para búsqueda de caminos
