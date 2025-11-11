"""
Script: train_maps_with_ai.py
Genera mapas de dungeon, bosque y camino usando la clase MapGenerator de la.py,
calcula métricas con MapMetrics y entrena un PreferenceModel con comparaciones
simuladas (o reales, si se desea pedir entrada del usuario).

Uso: python train_maps_with_ai.py
"""

import random
import numpy as np
from la import MapGenerator
from quality_agent.metrics import MapMetrics
from quality_agent.learning.preference_model import PreferenceModel
from quality_agent.learning.reward_calculator import RewardCalculator


def generate_map(env_type: str, overrides: dict | None = None):
    gen = MapGenerator(environment_type=env_type)
    if overrides:
        # aplicar overrides simples sobre la config interna
        for k, v in overrides.items():
            if k in gen.config:
                gen.config[k] = v
    grid = gen.generate()
    return grid


def to_vector(metrics_dict: dict, order: list[str]):
    return [float(metrics_dict[k]) for k in order]


def simulate_choice(vec_a, vec_b):
    # Regla de preferencia simple (ejemplo): mayor conectividad, luego menos dead_ends
    # Nota: dead_ends está invertida (1.0 es mejor). Ya está como “1 - ratio de callejones sin salida”
    score_a = vec_a[0] * 1.5 + vec_a[4] * 1.0
    score_b = vec_b[0] * 1.5 + vec_b[4] * 1.0
    if abs(score_a - score_b) < 1e-6:
        return random.choice(['A', 'B'])
    return 'A' if score_a > score_b else 'B'


def main(rounds: int = 5, ask_user: bool = False):
    metric_order = ['connectivity', 'density', 'room_distribution', 'corridor_quality', 'dead_ends']
    model = PreferenceModel(num_features=len(metric_order), critical_feature_index=0, critical_threshold=0.99)
    rewarder = RewardCalculator()
    comparisons = []  # (vec_A, vec_B, winner)

    env_pairs = [
        ('dungeon', 'dungeon'),
        ('forest', 'forest'),
        ('path_focused', 'path_focused'),
        # cruces entre tipos
        ('dungeon', 'forest'),
        ('forest', 'path_focused'),
        ('dungeon', 'path_focused'),
    ]

    for r in range(rounds):
        env1, env2 = random.choice(env_pairs)

        # Variar algunos parámetros de forma ligera por ronda
        overrides1 = {}
        overrides2 = {}
        if env1 == 'dungeon':
            overrides1 = {'min_leaf_size': random.randint(5, 9)}
        if env2 == 'dungeon':
            overrides2 = {'min_leaf_size': random.randint(5, 9)}
        if env1 == 'forest':
            overrides1 = {'forest_density': random.uniform(0.10, 0.20)}
        if env2 == 'forest':
            overrides2 = {'forest_density': random.uniform(0.10, 0.20)}
        if env1 == 'path_focused':
            overrides1 = {'path_width': random.randint(4, 7)}
        if env2 == 'path_focused':
            overrides2 = {'path_width': random.randint(4, 7)}

        grid_a = generate_map(env1, overrides1)
        grid_b = generate_map(env2, overrides2)

        metrics_a = MapMetrics(grid_a).as_dict()
        metrics_b = MapMetrics(grid_b).as_dict()
        vec_a = to_vector(metrics_a, metric_order)
        vec_b = to_vector(metrics_b, metric_order)

        if ask_user:
            print(f"Ronda {r+1}: ¿Cuál prefieres? A={env1} vs B={env2}")
            print(f"A: {metrics_a}")
            print(f"B: {metrics_b}")
            choice = input("Elige (A/B): ").strip().upper()
            while choice not in ('A', 'B'):
                choice = input("Elige válido (A/B): ").strip().upper()
        else:
            choice = simulate_choice(vec_a, vec_b)

        winner_flag = 1 if choice == 'A' else 0
        comparisons.append((vec_a, vec_b, winner_flag))

        # Recompensa (opcional, ilustrativo)
        feedback = vec_a if winner_flag == 1 else vec_b
        rew = rewarder.calculate_comparison_reward(feedback)
        print(f"[Ronda {r+1}] {env1} vs {env2} -> gana {choice}; recompensa={rew}")

    # Entrenar el modelo con todas las comparaciones
    model.learn_weights(comparisons)
    print("Pesos aprendidos:", model.weights)

    # Demo de predicción en los tres tipos
    for env in ['dungeon', 'forest', 'path_focused']:
        grid = generate_map(env)
        vec = to_vector(MapMetrics(grid).as_dict(), metric_order)
        # comparar contra dungeon base
        base = to_vector(MapMetrics(generate_map('dungeon')).as_dict(), metric_order)
        p = model.predict_preference(vec, base)
        print(f"Preferencia(model) {env} vs dungeon = {p:.3f}")


if __name__ == '__main__':
    # Cambia ask_user=True para elegir manualmente en cada ronda
    main(rounds=6, ask_user=False)
