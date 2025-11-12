"""
test.py - Archivo principal que orquesta el flujo de IA con Aprendizaje por Refuerzo
para Selección y Adaptación de Mapas

Flujo:
1. Genera N mapas usando la.py
2. Usa el evaluador para preseleccionar los 2 mejores mapas
3. Muestra los 2 mapas al usuario
4. Captura la elección del usuario (Mapa A o Mapa B)
5. Llama al adapter con el mapa ganador para aprender
"""

import numpy as np
from la import MapGenerator, print_map
from ia.evaluator import MapEvaluator
from ia.adapter import MapAdapter


def print_map_with_label(grid, label, environment_type):
    """Imprime un mapa con una etiqueta"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print_map(grid, environment_type)


def main(num_maps=10, environment_type='dungeon'):
    """
    Función principal que ejecuta el flujo completo
    
    Args:
        num_maps: Número de mapas a generar para selección
        environment_type: Tipo de entorno ('dungeon', 'forest', 'path_focused')
    """
    print("🎮 Sistema de IA con Aprendizaje por Refuerzo para Mapas")
    print(f"   Generando {num_maps} mapas de tipo: {environment_type}\n")
    
    # Inicializar evaluador y adaptador
    evaluator = MapEvaluator()
    adapter = MapAdapter()
    
    # Mostrar pesos actuales
    print("⚖️  Pesos actuales del evaluador:")
    for metric, weight in evaluator.weights.items():
        print(f"   {metric:20s}: {weight:.4f}")
    print()
    
    # Paso 1: Generar N mapas
    print(f"🔨 Generando {num_maps} mapas...")
    maps_data = []
    
    for i in range(num_maps):
        try:
            generator = MapGenerator(environment_type=environment_type)
            
            # Aplicar ajustes aprendidos si existen
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
            
            print(f"   Mapa {i+1:2d}: Score = {score:.4f}")
        except Exception as e:
            print(f"   ⚠️  Error generando mapa {i+1}: {e}")
            continue
    
    if len(maps_data) < 2:
        print("❌ No se pudieron generar suficientes mapas. Abortando.")
        return
    
    # Paso 2: Preseleccionar los 2 mejores mapas
    print(f"\n🎯 Preseleccionando los 2 mejores mapas...")
    maps_data.sort(key=lambda x: x['score'], reverse=True)
    top_2_maps = maps_data[:2]
    
    map_a = top_2_maps[0]
    map_b = top_2_maps[1]
    
    print(f"   Seleccionados: Mapa {map_a['id']} (Score: {map_a['score']:.4f}) y Mapa {map_b['id']} (Score: {map_b['score']:.4f})")
    
    # Paso 3: Mostrar los 2 mapas al usuario
    print_map_with_label(map_a['grid'], f"🅰️  MAPA A (ID: {map_a['id']})", environment_type)
    print("📊 Métricas del Mapa A:")
    for metric, value in map_a['metrics'].items():
        print(f"   {metric:20s}: {value:.4f}")
    
    print_map_with_label(map_b['grid'], f"🅱️  MAPA B (ID: {map_b['id']})", environment_type)
    print("📊 Métricas del Mapa B:")
    for metric, value in map_b['metrics'].items():
        print(f"   {metric:20s}: {value:.4f}")
    
    # Paso 4: Capturar la elección del usuario
    print(f"\n{'='*60}")
    print("❓ ¿Cuál mapa prefieres?")
    print("   Escribe 'A' para el Mapa A")
    print("   Escribe 'B' para el Mapa B")
    print("   Escribe 'Q' para salir sin aprender")
    print(f"{'='*60}")
    
    choice = input("\n👉 Tu elección (A/B/Q): ").strip().upper()
    
    while choice not in ['A', 'B', 'Q']:
        print("⚠️  Opción inválida. Por favor elige 'A', 'B' o 'Q'.")
        choice = input("👉 Tu elección (A/B/Q): ").strip().upper()
    
    if choice == 'Q':
        print("👋 Saliendo sin aprender. ¡Hasta luego!")
        return
    
    # Paso 5: Aprender de la elección
    winning_map = map_a if choice == 'A' else map_b
    losing_map = map_b if choice == 'A' else map_a
    
    print(f"\n✅ Elegiste el Mapa {choice}!")
    print("\n🧠 Aprendiendo de tu elección...")
    
    # Aprender ajustando pesos
    new_weights = adapter.learn(
        winning_map['metrics'],
        losing_map['metrics']
    )
    
    # Ajustar parámetros del entorno
    print(f"\n🔧 Ajustando parámetros del entorno '{environment_type}'...")
    env_adjustments = adapter.adjust_environment_params(
        environment_type,
        winning_map['metrics']
    )
    
    # Recargar pesos en el evaluador
    evaluator.reload_weights()
    
    print("\n✨ ¡Aprendizaje completado!")
    print("   El sistema ahora está mejor adaptado a tus preferencias.")
    print("   Ejecuta este script nuevamente para ver mapas mejorados.\n")


def interactive_loop():
    """Bucle interactivo que permite múltiples rondas de aprendizaje"""
    print("🎮 Modo Interactivo - Sistema de IA para Mapas")
    print("="*60)
    
    while True:
        print("\n📋 Opciones:")
        print("   1. Entrenar con mapas de Dungeon")
        print("   2. Entrenar con mapas de Forest")
        print("   3. Entrenar con mapas de Path-Focused")
        print("   4. Entrenar con tipo mixto (aleatorio)")
        print("   5. Ver pesos actuales")
        print("   6. Resetear pesos y ajustes")
        print("   Q. Salir")
        
        option = input("\n👉 Elige una opción: ").strip().upper()
        
        if option == 'Q':
            print("👋 ¡Hasta luego!")
            break
        elif option == '1':
            main(num_maps=10, environment_type='dungeon')
        elif option == '2':
            main(num_maps=10, environment_type='forest')
        elif option == '3':
            main(num_maps=10, environment_type='path_focused')
        elif option == '4':
            import random
            env_type = random.choice(['dungeon', 'forest', 'path_focused'])
            print(f"🎲 Tipo seleccionado aleatoriamente: {env_type}")
            main(num_maps=10, environment_type=env_type)
        elif option == '5':
            show_current_weights()
        elif option == '6':
            reset_system()
        else:
            print("⚠️  Opción inválida.")


def show_current_weights():
    """Muestra los pesos actuales del sistema"""
    import json
    from pathlib import Path
    
    weights_path = Path(__file__).parent / "ia" / "configs" / "weights.json"
    env_path = Path(__file__).parent / "ia" / "configs" / "environment_adjustments.json"
    
    print("\n" + "="*60)
    print("📊 Estado actual del sistema")
    print("="*60)
    
    try:
        with open(weights_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n⚖️  Pesos del evaluador (Iteración {data.get('iteration', 0)}):")
        for metric, weight in data.get('weights', {}).items():
            print(f"   {metric:20s}: {weight:.4f}")
    except Exception as e:
        print(f"⚠️  Error leyendo pesos: {e}")
    
    try:
        with open(env_path, 'r') as f:
            adjustments = json.load(f)
        
        if adjustments:
            print(f"\n🔧 Ajustes de entorno:")
            for env_type, params in adjustments.items():
                print(f"\n   {env_type}:")
                for param, value in params.items():
                    print(f"      {param}: {value}")
        else:
            print("\n🔧 No hay ajustes de entorno aún.")
    except Exception as e:
        print(f"⚠️  Error leyendo ajustes: {e}")
    
    print()


def reset_system():
    """Resetea el sistema a sus valores por defecto"""
    import json
    from pathlib import Path
    
    weights_path = Path(__file__).parent / "ia" / "configs" / "weights.json"
    env_path = Path(__file__).parent / "ia" / "configs" / "environment_adjustments.json"
    
    print("\n⚠️  ¿Estás seguro de que quieres resetear el sistema?")
    print("   Esto eliminará todo el aprendizaje previo.")
    confirm = input("👉 Escribe 'SI' para confirmar: ").strip().upper()
    
    if confirm == 'SI':
        # Resetear pesos
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
        
        # Resetear ajustes de entorno
        with open(env_path, 'w') as f:
            json.dump({}, f, indent=2)
        
        print("✅ Sistema reseteado a valores por defecto.")
    else:
        print("❌ Operación cancelada.")


if __name__ == "__main__":
    # Ejecutar en modo interactivo
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
