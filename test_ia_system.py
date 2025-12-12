"""
Test automatizado para validar el sistema de IA
Simula la interacción del usuario para verificar que todo funciona correctamente
"""

import sys
import numpy as np
from la import MapGenerator
from ia.evaluator import MapEvaluator
from ia.adapter import MapAdapter


def test_evaluator():
    """Prueba el evaluador de mapas"""
    print("🧪 Probando MapEvaluator...")
    
    evaluator = MapEvaluator()
    
    # Crear un mapa de prueba simple
    test_map = np.zeros((30, 50), dtype=int)
    test_map[10:20, 10:40] = 1  # Una habitación grande
    
    score, metrics = evaluator.score(test_map)
    
    assert score > 0, "El score debe ser mayor que 0"
    assert 'room_density' in metrics, "Debe calcular room_density"
    assert 'path_density' in metrics, "Debe calcular path_density"
    assert 'obstacle_density' in metrics, "Debe calcular obstacle_density"
    assert 'avg_room_size' in metrics, "Debe calcular avg_room_size"
    assert 'connectivity' in metrics, "Debe calcular connectivity"
    
    print("   ✅ MapEvaluator funciona correctamente")
    print(f"   Score obtenido: {score:.4f}")
    return True


def test_adapter():
    """Prueba el adaptador de pesos"""
    print("\n🧪 Probando MapAdapter...")
    
    adapter = MapAdapter()
    
    # Métricas simuladas
    winning_metrics = {
        'room_density': 0.35,
        'path_density': 0.30,
        'obstacle_density': 0.10,
        'avg_room_size': 0.45,
        'connectivity': 0.95
    }
    
    losing_metrics = {
        'room_density': 0.25,
        'path_density': 0.40,
        'obstacle_density': 0.15,
        'avg_room_size': 0.25,
        'connectivity': 0.85
    }
    
    # Aprender de la elección
    new_weights = adapter.learn(winning_metrics, losing_metrics)
    
    assert isinstance(new_weights, dict), "Debe devolver un diccionario de pesos"
    assert len(new_weights) == 5, "Debe tener 5 pesos"
    
    # Verificar que los pesos suman aproximadamente 1.0
    total = sum(new_weights.values())
    assert 0.99 < total < 1.01, f"Los pesos deben sumar ~1.0, suma actual: {total}"
    
    print("   ✅ MapAdapter funciona correctamente")
    print(f"   Nuevos pesos: {new_weights}")
    return True


def test_map_generation_and_evaluation():
    """Prueba la generación y evaluación de mapas reales"""
    print("\n🧪 Probando generación y evaluación de mapas reales...")
    
    evaluator = MapEvaluator()
    
    for env_type in ['dungeon', 'forest', 'path_focused']:
        try:
            print(f"\n   Probando entorno: {env_type}")
            generator = MapGenerator(environment_type=env_type)
            map_grid = generator.generate()
            
            score, metrics = evaluator.score(map_grid)
            
            assert map_grid is not None, f"Debe generar un mapa para {env_type}"
            assert score >= 0, f"El score debe ser no negativo para {env_type}"
            
            print(f"   ✅ {env_type}: Score = {score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error con {env_type}: {e}")
            return False
    
    return True


def test_full_workflow():
    """Prueba el flujo completo con selección automática"""
    print("\n🧪 Probando flujo completo del sistema...")
    
    num_maps = 5
    environment_type = 'dungeon'
    
    evaluator = MapEvaluator()
    adapter = MapAdapter()
    
    # Generar mapas
    print(f"\n   Generando {num_maps} mapas de tipo {environment_type}...")
    maps_data = []
    
    for i in range(num_maps):
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
            
        except Exception as e:
            print(f"   ⚠️  Error generando mapa {i+1}: {e}")
            continue
    
    assert len(maps_data) >= 2, "Debe generar al menos 2 mapas"
    
    # Preseleccionar los 2 mejores
    maps_data.sort(key=lambda x: x['score'], reverse=True)
    top_2_maps = maps_data[:2]
    
    map_a = top_2_maps[0]
    map_b = top_2_maps[1]
    
    print(f"   Top 2 mapas seleccionados:")
    print(f"      Mapa A: Score = {map_a['score']:.4f}")
    print(f"      Mapa B: Score = {map_b['score']:.4f}")
    
    # Simular elección del usuario (siempre elige el de mayor score)
    winning_map = map_a
    losing_map = map_b
    
    print(f"\n   Simulando elección del usuario (Mapa A)...")
    
    # Aprender
    new_weights = adapter.learn(winning_map['metrics'], losing_map['metrics'])
    
    assert new_weights is not None, "Debe devolver nuevos pesos"
    
    # Ajustar parámetros del entorno
    env_adjustments = adapter.adjust_environment_params(
        environment_type,
        winning_map['metrics']
    )
    
    print("   ✅ Flujo completo ejecutado correctamente")
    return True


def test_environment_adjustments():
    """Prueba los ajustes de entorno"""
    print("\n🧪 Probando ajustes de entorno...")
    
    adapter = MapAdapter()
    
    # Métricas que deberían provocar ajustes
    high_density_metrics = {
        'room_density': 0.5,
        'path_density': 0.45,
        'obstacle_density': 0.20,
        'avg_room_size': 0.70,
        'connectivity': 0.95
    }
    
    for env_type in ['dungeon', 'forest', 'path_focused']:
        adjustments = adapter.adjust_environment_params(env_type, high_density_metrics)
        print(f"   ✅ {env_type}: {len(adjustments)} ajustes aplicados")
    
    return True


def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("="*60)
    print("🚀 Ejecutando suite de pruebas del sistema de IA")
    print("="*60)
    
    tests = [
        ("Evaluador", test_evaluator),
        ("Adaptador", test_adapter),
        ("Generación y Evaluación", test_map_generation_and_evaluation),
        ("Ajustes de Entorno", test_environment_adjustments),
        ("Flujo Completo", test_full_workflow),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n   ❌ {test_name} falló")
        except Exception as e:
            failed += 1
            print(f"\n   ❌ {test_name} falló con error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"📊 Resultados: {passed} pasadas, {failed} fallidas")
    print("="*60)
    
    if failed == 0:
        print("✅ ¡Todas las pruebas pasaron exitosamente!")
        return 0
    else:
        print("❌ Algunas pruebas fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
