# test_learning_agent.py
from generator import create_dungeon_layout
from quality_agent.storage import PreferenceStorage
from quality_agent.learning import PreferenceModel, RewardCalculator
from quality_agent.metrics import MapMetrics

# Configuración del generador
config_a = {
    'width': 50,
    'height': 30,
    'min_leaf_size': 6,
    'padding': 1
}

config_b = {
    'width': 50,
    'height': 30,
    'min_leaf_size': 8,  # Habitaciones más grandes
    'padding': 2          # Más separación
}

print("🎮 Generando Mapa A...")
map_a = create_dungeon_layout(config_a)

print("🎮 Generando Mapa B...")
map_b = create_dungeon_layout(config_b)

metrics_a = MapMetrics(map_a).as_dict()
metrics_b = MapMetrics(map_b).as_dict()
metric_order = ['connectivity', 'density', 'room_distribution', 'corridor_quality', 'dead_ends']
vector_a = [metrics_a[name] for name in metric_order]
vector_b = [metrics_b[name] for name in metric_order]

# Visualizar ambos mapas
print("\n📍 MAPA A:")
for y in range(map_a.shape[0]):
    line = ""
    for x in range(map_a.shape[1]):
        line += "#" if map_a[y, x] == 0 else "."
    print(line)

print("\n📍 MAPA B:")
# Mostrar métricas calculadas
print("\n📐 Métricas calculadas para el Mapa A:")
for name in metric_order:
    print(f"  - {name}: {metrics_a[name]:.4f}")

print("\n📐 Métricas calculadas para el Mapa B:")
for name in metric_order:
    print(f"  - {name}: {metrics_b[name]:.4f}")
for y in range(map_b.shape[0]):
    line = ""
    for x in range(map_b.shape[1]):
        line += "#" if map_b[y, x] == 0 else "."
    print(line)

# === Mapas adicionales: Bosque y Camino ===
config_forest = {
    'width': 50,
    'height': 30,
    'environment_type': 'forest'
}

config_path = {
    'width': 50,
    'height': 30,
    'environment_type': 'path_focused'
}

print("\n🌲 Generando Mapa Bosque (F)...")
map_forest = create_dungeon_layout(config_forest)
metrics_forest = MapMetrics(map_forest).as_dict()
vector_forest = [metrics_forest[name] for name in metric_order]

print("🛤️ Generando Mapa Camino (P)...")
map_path = create_dungeon_layout(config_path)
metrics_path = MapMetrics(map_path).as_dict()
vector_path = [metrics_path[name] for name in metric_order]

print("\n📍 MAPA BOSQUE (F):")
for y in range(map_forest.shape[0]):
    line = ""
    for x in range(map_forest.shape[1]):
        val = map_forest[y, x]
        line += "#" if val == 0 else ("." if val == 1 else "O")
    print(line)

print("\n📐 Métricas Bosque:")
for name in metric_order:
    print(f"  - {name}: {metrics_forest[name]:.4f}")

print("\n📍 MAPA CAMINO (P):")
for y in range(map_path.shape[0]):
    line = ""
    for x in range(map_path.shape[1]):
        val = map_path[y, x]
        line += "#" if val == 0 else ("." if val == 1 else "O")
    print(line)

print("\n📐 Métricas Camino:")
for name in metric_order:
    print(f"  - {name}: {metrics_path[name]:.4f}")

# Simular preferencia del usuario
print("\n❓ ¿Cuál mapa prefieres? (A/B)")
user_choice = input("Tu elección: ").upper()

# Guardar la preferencia
storage = PreferenceStorage()
user_id = "test_user_1"

comparison_data = {
    "map_a": {
        "config": config_a,
    "metrics": {name: {"score": round(metrics_a[name], 4)} for name in metric_order}
    },
    "map_b": {
        "config": config_b,
    "metrics": {name: {"score": round(metrics_b[name], 4)} for name in metric_order}
    },
    "preferred": user_choice
}

storage.save_comparison(user_id, comparison_data)

print(f"\n✅ Preferencia guardada! Total de comparaciones: {len(storage.get_user_comparisons(user_id))}")

# Calcular recompensa
reward_calc = RewardCalculator()
feedback_data = vector_a if user_choice == "A" else vector_b
reward = reward_calc.calculate_comparison_reward(feedback_data)
print(f"🎁 Recompensa calculada: {reward}")

# Segunda comparación: Bosque vs Camino
print("\n❓ ¿Cuál mapa prefieres entre Bosque y Camino? (F/P)")
user_choice_fp = input("Tu elección (F/P): ").upper()
while user_choice_fp not in ("F", "P"):
    user_choice_fp = input("Por favor, elige 'F' (Bosque) o 'P' (Camino): ").upper()

comparison_data_fp = {
    "map_a": {
        "config": config_forest,
        "metrics": {name: {"score": round(metrics_forest[name], 4)} for name in metric_order}
    },
    "map_b": {
        "config": config_path,
        "metrics": {name: {"score": round(metrics_path[name], 4)} for name in metric_order}
    },
    "preferred": "A" if user_choice_fp == "F" else "B"
}
storage.save_comparison(user_id, comparison_data_fp)

print(f"\n✅ Preferencia guardada! Total de comparaciones: {len(storage.get_user_comparisons(user_id))}")

feedback_data_fp = vector_forest if user_choice_fp == "F" else vector_path
reward_fp = reward_calc.calculate_comparison_reward(feedback_data_fp)
print(f"🎁 Recompensa calculada (Bosque/Camino): {reward_fp}")

# Si hay suficientes comparaciones, aprender
def _comparisons_to_training_data(comparisons, metric_order):
    data = []
    for c in comparisons:
        mA = c["map_a"]["metrics"]
        mB = c["map_b"]["metrics"]
        vecA = [float(mA[k]["score"]) for k in metric_order]
        vecB = [float(mB[k]["score"]) for k in metric_order]
        winner = 1 if c["preferred"].upper() == "A" else 0
        data.append((vecA, vecB, winner))
    return data

comparisons_needed = 2  # entrenar con 2 comparaciones en esta demo
if len(storage.get_user_comparisons(user_id)) >= comparisons_needed:
    print("\n🧠 Aprendiendo preferencias del usuario...")
    model = PreferenceModel(num_features=len(metric_order), critical_feature_index=0, critical_threshold=0.7)
    comparisons_raw = storage.get_user_comparisons(user_id)
    training_data = _comparisons_to_training_data(comparisons_raw, metric_order)
    model.learn_weights(training_data)
    print("📈 Pesos aprendidos:", [round(w, 4) for w in model.weights])
    print("✨ ¡Sistema listo para personalizar mapas!")
else:
    print(f"\n📊 Necesitas {comparisons_needed - len(storage.get_user_comparisons(user_id))} comparaciones más para que el sistema aprenda tus preferencias")
