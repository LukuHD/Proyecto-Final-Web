from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import PlainTextResponse # <-- IMPORTANTE: Importar esto

from generator import create_dungeon_layout

#figuras
from figuras import GestorFiguras

app = FastAPI(
    title="Dungeon Generator API",
    description="Una API para generar proceduralmente mapas de calabozos.",
    version="0.1.0",
)
class MapConfig(BaseModel):
    width: int = 50
    height: int = 30
    min_leaf_size: int = 6
    padding: Optional[int] = 1

#para las figuras
gestor = GestorFiguras(limite=5)
# Supongamos que el usuario selecciona desde la checklist:
seleccion_usuario = ["circulo", "elipse", "rectangulo"]
for nombre in seleccion_usuario:
    gestor.agregar_figura(nombre)

mapa_final = gestor.generar(80, 80)



@app.post("/generate_map/")
def generate_map_endpoint(config: MapConfig):
    config_dict = config.dict()
    dungeon_grid = create_dungeon_layout(config_dict)
    return {"map_layout": dungeon_grid.tolist()}

# --- NUEVO ENDPOINT DE PRUEBA ---
@app.post("/generate_map/visual_text/", response_class=PlainTextResponse)
def generate_map_visual_text(config: MapConfig):
    """
    Genera un mapa y lo devuelve como una cadena de texto para visualización rápida.
    """
    config_dict = config.dict()
    dungeon_grid = create_dungeon_layout(config_dict)
    
    # Convertir la cuadrícula de NumPy a una cadena de texto
    map_string = ""
    height, width = dungeon_grid.shape
    for y in range(height):
        for x in range(width):
            map_string += "#" if dungeon_grid[y, x] == 0 else "."
        map_string += "\n" # Añadir un salto de línea al final de cada fila
        
    return map_string