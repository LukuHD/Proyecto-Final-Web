"""
MapAdapter: Aprende de las elecciones del usuario y ajusta pesos del evaluator
Implementa aprendizaje por refuerzo para adaptarse a las preferencias del usuario
"""

import json
import os
import numpy as np
from pathlib import Path
from copy import deepcopy


class MapAdapter:
    """
    Adapta los pesos del evaluador basándose en las elecciones del usuario
    Usa aprendizaje por refuerzo simple para actualizar preferencias
    """
    
    def __init__(self, weights_config_path=None, env_config_path=None, learning_rate=0.1):
        """
        Inicializa el adaptador
        
        Args:
            weights_config_path: Ruta al archivo weights.json
            env_config_path: Ruta al archivo environment_adjustments.json
            learning_rate: Tasa de aprendizaje (0.0 a 1.0)
        """
        if weights_config_path is None:
            weights_config_path = Path(__file__).parent / "configs" / "weights.json"
        if env_config_path is None:
            env_config_path = Path(__file__).parent / "configs" / "environment_adjustments.json"
        
        self.weights_config_path = weights_config_path
        self.env_config_path = env_config_path
        self.learning_rate = learning_rate
        self.iteration_count = 0
    
    def _load_weights(self):
        """Carga los pesos actuales desde el archivo"""
        try:
            with open(self.weights_config_path, 'r') as f:
                data = json.load(f)
                return data.get('weights', {}), data.get('iteration', 0)
        except FileNotFoundError:
            return {}, 0
        except json.JSONDecodeError:
            return {}, 0
    
    def _save_weights(self, weights, iteration):
        """Guarda los pesos actualizados en el archivo"""
        data = {
            'weights': weights,
            'iteration': iteration,
            'learning_rate': self.learning_rate
        }
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(self.weights_config_path), exist_ok=True)
        
        with open(self.weights_config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_env_adjustments(self):
        """Carga los ajustes de entorno actuales"""
        try:
            with open(self.env_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _save_env_adjustments(self, adjustments):
        """Guarda los ajustes de entorno"""
        os.makedirs(os.path.dirname(self.env_config_path), exist_ok=True)
        
        with open(self.env_config_path, 'w') as f:
            json.dump(adjustments, f, indent=2)
    
    def learn(self, winning_map_metrics, losing_map_metrics=None):
        """
        Aprende de la elección del usuario y ajusta los pesos
        
        Estrategia de aprendizaje:
        - Incrementa los pesos de las métricas que son más altas en el mapa ganador
        - Si hay un mapa perdedor, decrementan ligeramente los pesos de sus métricas dominantes
        - Normaliza los pesos para que sumen 1.0
        
        Args:
            winning_map_metrics: dict con las métricas del mapa ganador
            losing_map_metrics: dict con las métricas del mapa perdedor (opcional)
        
        Returns:
            dict: Los nuevos pesos después del aprendizaje
        """
        # Cargar pesos actuales
        current_weights, iteration = self._load_weights()
        
        if not current_weights:
            # Inicializar con pesos balanceados
            current_weights = {
                'room_density': 0.25,
                'path_density': 0.25,
                'obstacle_density': 0.15,
                'avg_room_size': 0.20,
                'connectivity': 0.15
            }
        
        # Crear copia para modificar
        new_weights = deepcopy(current_weights)
        
        # Estrategia 1: Reforzar características del mapa ganador
        # Incrementar pesos proporcionalmente a las métricas del ganador
        for metric, value in winning_map_metrics.items():
            if metric in new_weights:
                # Incremento proporcional al valor de la métrica y la tasa de aprendizaje
                increment = self.learning_rate * value * 0.5
                new_weights[metric] += increment
        
        # Estrategia 2: Si hay mapa perdedor, penalizar sus características dominantes
        if losing_map_metrics:
            for metric, value in losing_map_metrics.items():
                if metric in new_weights:
                    # Si el perdedor tenía esta métrica más alta que el ganador
                    if value > winning_map_metrics.get(metric, 0):
                        decrement = self.learning_rate * (value - winning_map_metrics[metric]) * 0.3
                        new_weights[metric] = max(0.05, new_weights[metric] - decrement)  # Mínimo 0.05
        
        # Normalizar pesos para que sumen 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        # Asegurar que ningún peso sea demasiado pequeño o grande
        for key in new_weights:
            new_weights[key] = max(0.05, min(0.50, new_weights[key]))
        
        # Re-normalizar después de aplicar límites
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        # Guardar nuevos pesos
        iteration += 1
        self._save_weights(new_weights, iteration)
        
        print(f"✨ Aprendizaje completado (iteración {iteration})")
        print(f"📊 Nuevos pesos:")
        for metric, weight in sorted(new_weights.items()):
            change = new_weights[metric] - current_weights.get(metric, 0)
            arrow = "↑" if change > 0 else "↓" if change < 0 else "="
            print(f"   {metric:20s}: {weight:.4f} {arrow}")
        
        return new_weights
    
    def adjust_environment_params(self, environment_type, winning_map_metrics):
        """
        Ajusta los parámetros de generación de mapas basándose en las métricas del ganador
        
        Args:
            environment_type: Tipo de entorno ('dungeon', 'forest', 'path_focused')
            winning_map_metrics: Métricas del mapa ganador
        
        Returns:
            dict: Ajustes aplicados
        """
        adjustments = self._load_env_adjustments()
        
        if environment_type not in adjustments:
            adjustments[environment_type] = {}
        
        env_adjustments = adjustments[environment_type]
        
        # Ajustar parámetros basándose en las métricas del ganador
        
        # 1. Densidad de habitaciones/caminos -> ajustar tamaño de mapa o min_leaf_size
        if winning_map_metrics.get('room_density', 0) > 0.4:
            # Usuario prefiere mapas más densos
            if environment_type == 'dungeon':
                env_adjustments['min_leaf_size'] = max(4, env_adjustments.get('min_leaf_size', 6) - 1)
        elif winning_map_metrics.get('room_density', 0) < 0.25:
            # Usuario prefiere mapas menos densos
            if environment_type == 'dungeon':
                env_adjustments['min_leaf_size'] = min(12, env_adjustments.get('min_leaf_size', 6) + 1)
        
        # 2. Obstáculos -> ajustar cantidad de obstáculos
        if winning_map_metrics.get('obstacle_density', 0) > 0.15:
            # Usuario prefiere más obstáculos
            env_adjustments['obstacle_multiplier'] = min(2.0, env_adjustments.get('obstacle_multiplier', 1.0) + 0.1)
        elif winning_map_metrics.get('obstacle_density', 0) < 0.05:
            # Usuario prefiere menos obstáculos
            env_adjustments['obstacle_multiplier'] = max(0.3, env_adjustments.get('obstacle_multiplier', 1.0) - 0.1)
        
        # 3. Tamaño de habitaciones -> ajustar room_min_size para dungeons
        if environment_type == 'dungeon':
            if winning_map_metrics.get('avg_room_size', 0) > 0.6:
                env_adjustments['room_min_size'] = min(8, env_adjustments.get('room_min_size', 4) + 1)
            elif winning_map_metrics.get('avg_room_size', 0) < 0.3:
                env_adjustments['room_min_size'] = max(3, env_adjustments.get('room_min_size', 4) - 1)
        
        # 4. Caminos para forest
        if environment_type == 'forest':
            if winning_map_metrics.get('path_density', 0) > 0.4:
                env_adjustments['path_width'] = min(4, env_adjustments.get('path_width', 2) + 1)
            elif winning_map_metrics.get('path_density', 0) < 0.2:
                env_adjustments['path_width'] = max(1, env_adjustments.get('path_width', 2) - 1)
        
        # 5. Conectividad -> ajustar extra_path_connections_prob
        if environment_type == 'forest':
            if winning_map_metrics.get('connectivity', 0) > 0.9:
                env_adjustments['extra_path_connections_prob'] = min(0.5, 
                    env_adjustments.get('extra_path_connections_prob', 0.25) + 0.05)
            elif winning_map_metrics.get('connectivity', 0) < 0.8:
                env_adjustments['extra_path_connections_prob'] = max(0.1, 
                    env_adjustments.get('extra_path_connections_prob', 0.25) - 0.05)
        
        # Guardar ajustes
        adjustments[environment_type] = env_adjustments
        self._save_env_adjustments(adjustments)
        
        print(f"🔧 Ajustes de entorno para '{environment_type}':")
        for param, value in env_adjustments.items():
            print(f"   {param}: {value}")
        
        return env_adjustments
    
    def get_adjusted_config(self, base_config, environment_type):
        """
        Obtiene la configuración ajustada para un tipo de entorno
        
        Args:
            base_config: Configuración base del entorno
            environment_type: Tipo de entorno
        
        Returns:
            dict: Configuración ajustada
        """
        adjustments = self._load_env_adjustments()
        env_adjustments = adjustments.get(environment_type, {})
        
        # Aplicar ajustes sobre la configuración base
        adjusted_config = deepcopy(base_config)
        
        for key, value in env_adjustments.items():
            if key == 'obstacle_multiplier':
                # Aplicar multiplicador a obstáculos
                if 'obstacles' in adjusted_config:
                    for obstacle in adjusted_config['obstacles']:
                        if 'count' in obstacle:
                            obstacle['count'] = max(1, int(obstacle['count'] * value))
            else:
                # Aplicar directamente otros ajustes
                adjusted_config[key] = value
        
        return adjusted_config
