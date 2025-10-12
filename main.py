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


"""
from fastapi import FastAPI
from pydantic import BaseModel
from figuras import GestorFiguras
import numpy as np

app = FastAPI(title="Generador de Mapas con Figuras")

# =====================================================
# MODELOS DE ENTRADA
# =====================================================
class MapRequest(BaseModel):
    ancho: int = 80
    alto: int = 80
    figuras: list[str] = []  # Lista con nombres de figuras seleccionadas
    limite: int = 5           # Máximo de figuras permitidas


# =====================================================
# RUTAS DE LA API
# =====================================================
@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido al generador de mapas con figuras 🗺️",
        "instrucciones": "Usa POST /generar_mapa con una lista de figuras."
    }


@app.post("/generar_mapa")
def generar_mapa(data: MapRequest):
    gestor = GestorFiguras(limite=data.limite)

    # Agregar las figuras seleccionadas por el usuario
    for nombre in data.figuras:
        gestor.agregar_figura(nombre.lower())

    # Generar el mapa
    mapa = gestor.generar(data.ancho, data.alto)

    # Convertir el mapa en una lista (para JSON)
    mapa_lista = mapa.tolist()

    return {
        "ancho": data.ancho,
        "alto": data.alto,
        "figuras_generadas": [f.nombre for f in gestor.registro],
        "mapa": mapa_lista
    }


@app.post("/generar_mapa/visual_text")
def generar_mapa_texto(data: MapRequest):
    gestor = GestorFiguras(limite=data.limite)

    for nombre in data.figuras:
        gestor.agregar_figura(nombre.lower())

    mapa = gestor.generar(data.ancho, data.alto)

    # Convertir matriz en texto
    texto = ""
    for fila in mapa:
        texto += "".join(["#" if x == 1 else "." for x in fila]) + "\n"

    return {
        "ancho": data.ancho,
        "alto": data.alto,
        "figuras_generadas": [f.nombre for f in gestor.registro],
        "mapa_texto": texto
    }
"""