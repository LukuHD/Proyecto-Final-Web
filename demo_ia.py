"""
Demo rápido del sistema de IA - Ejecuta un ciclo completo de aprendizaje
Este script demuestra el sistema sin requerir interacción del usuario
"""

import numpy as np
from la import MapGenerator, print_map
from ia.evaluator import MapEvaluator
from ia.adapter import MapAdapter


def demo_quick_learning():
    """Demostración rápida del sistema de aprendizaje"""
    print("="*70)
    print("🎮 DEMO: Sistema de IA con Aprendizaje por Refuerzo")
    print("="*70)
    
    # Resetear sistema para la demo
    import json
    from pathlib import Path
    
    weights_path = Path(__file__).parent / "ia" / "configs" / "weights.json"
    default_weights = {
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
    with open(weights_path, 'w') as f:
        json.dump(default_weights, f, indent=2)
    
    print("\n🔄 Sistema reseteado a valores por defecto")
    
    # Inicializar
    evaluator = MapEvaluator()
    adapter = MapAdapter()
    environment_type = 'dungeon'
    
    print(f"\n⚖️  Pesos iniciales:")
    for metric, weight in evaluator.weights.items():
        print(f"   {metric:20s}: {weight:.4f}")
    
    # Simular 3 rondas de aprendizaje
    for round_num in range(1, 4):
        print(f"\n{'='*70}")
        print(f"🔄 RONDA {round_num} DE APRENDIZAJE")
        print(f"{'='*70}")
        
        # Generar 5 mapas
        print(f"\n🔨 Generando 5 mapas de tipo '{environment_type}'...")
        maps_data = []
        
        for i in range(5):
            try:
                generator = MapGenerator(environment_type=environment_type)
                adjusted_config = adapter.get_adjusted_config(generator.config, environment_type)
                generator.config = adjusted_config
                
                map_grid = generator.generate()
                score, metrics = evaluator.score(map_grid)
                
                maps_data.append({
                    'id': i + 1,
                    'grid': map_grid,
                    'score': score,
                    'metrics': metrics
                })
                
                print(f"   Mapa {i+1}: Score = {score:.4f}")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
                continue
        
        if len(maps_data) < 2:
            print("❌ No hay suficientes mapas")
            break
        
        # Seleccionar los 2 mejores
        maps_data.sort(key=lambda x: x['score'], reverse=True)
        map_a = maps_data[0]
        map_b = maps_data[1]
        
        print(f"\n🎯 Top 2 mapas:")
        print(f"   Mapa A (ID {map_a['id']}): Score = {map_a['score']:.4f}")
        print(f"   Mapa B (ID {map_b['id']}): Score = {map_b['score']:.4f}")
        
        # Mostrar solo el ganador (el de mayor score)
        print(f"\n📍 Mapa ganador (Mapa A):")
        print_map(map_a['grid'], environment_type)
        
        print("\n📊 Métricas del ganador:")
        for metric, value in map_a['metrics'].items():
            print(f"   {metric:20s}: {value:.4f}")
        
        # Simular elección (siempre el de mayor score)
        print(f"\n🤖 Sistema elige automáticamente: Mapa A (mayor score)")
        
        # Aprender
        print(f"\n🧠 Aprendiendo de la elección...")
        new_weights = adapter.learn(map_a['metrics'], map_b['metrics'])
        
        # Ajustar parámetros
        env_adjustments = adapter.adjust_environment_params(
            environment_type,
            map_a['metrics']
        )
        
        # Recargar pesos
        evaluator.reload_weights()
    
    print(f"\n{'='*70}")
    print("✨ DEMO COMPLETADA")
    print(f"{'='*70}")
    
    print(f"\n⚖️  Pesos finales después de 3 rondas:")
    for metric, weight in evaluator.weights.items():
        print(f"   {metric:20s}: {weight:.4f}")
    
    print("\n📈 El sistema ha aprendido y ajustado los pesos basándose en los mapas")
    print("   preferidos. Los próximos mapas generados reflejarán estas preferencias.\n")


if __name__ == "__main__":
    try:
        demo_quick_learning()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrumpida")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
