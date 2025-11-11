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

# Si hay suficientes comparaciones, aprender
if len(storage.get_user_comparisons(user_id)) >= 5:
    print("\n🧠 Aprendiendo preferencias del usuario...")
    model = PreferenceModel(num_features=len(metric_order), critical_feature_index=0, critical_threshold=0.99)
    comparisons = storage.get_user_comparisons(user_id)
    
    # Aquí se aprendería los pesos (necesitaríamos más datos reales)
    print("✨ ¡Sistema listo para personalizar mapas!")
else:
    print(f"\n📊 Necesitas {5 - len(storage.get_user_comparisons(user_id))} comparaciones más para que el sistema aprenda tus preferencias")
