# test_learning_agent.py
from generator import create_dungeon_layout
from quality_agent.storage import PreferenceStorage
from quality_agent.learning import PreferenceModel, FeedbackCollector, RewardCalculator
import numpy as np

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

# Visualizar ambos mapas
print("\n📍 MAPA A:")
for y in range(map_a.shape[0]):
    line = ""
    for x in range(map_a.shape[1]):
        line += "#" if map_a[y, x] == 0 else "."
    print(line)

print("\n📍 MAPA B:")
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
        "metrics": {
            "connectivity": {"score": 85},
            "density": {"score": 70},
            "room_distribution": {"score": 75},
            "corridor_quality": {"score": 80},
            "dead_ends": {"score": 90}
        }
    },
    "map_b": {
        "config": config_b,
        "metrics": {
            "connectivity": {"score": 90},
            "density": {"score": 65},
            "room_distribution": {"score": 80},
            "corridor_quality": {"score": 75},
            "dead_ends": {"score": 85}
        }
    },
    "preferred": user_choice
}

storage.save_comparison(user_id, comparison_data)

print(f"\n✅ Preferencia guardada! Total de comparaciones: {len(storage.get_user_comparisons(user_id))}")

# Calcular recompensa
reward_calc = RewardCalculator()
feedback_data = [85, 70, 75, 80, 90] if user_choice == "A" else [90, 65, 80, 75, 85]
reward = reward_calc.calculate_comparison_reward(feedback_data)
print(f"🎁 Recompensa calculada: {reward}")

# Si hay suficientes comparaciones, aprender
if len(storage.get_user_comparisons(user_id)) >= 5:
    print("\n🧠 Aprendiendo preferencias del usuario...")
    model = PreferenceModel(num_features=5)
    comparisons = storage.get_user_comparisons(user_id)
    
    # Aquí se aprendería los pesos (necesitaríamos más datos reales)
    print("✨ ¡Sistema listo para personalizar mapas!")
else:
    print(f"\n📊 Necesitas {5 - len(storage.get_user_comparisons(user_id))} comparaciones más para que el sistema aprenda tus preferencias")
