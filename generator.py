import random
import numpy as np

# --- Clases y Funciones Auxiliares ---

class Leaf:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.child_1 = None
        self.child_2 = None
        self.room = None

    def split(self, min_leaf_size):
        # MODIFICADO: min_leaf_size ahora es un argumento
        if self.child_1 is not None or self.child_2 is not None:
            return False

        split_horizontally = random.choice([True, False])
        if self.width > self.height and self.width / self.height >= 1.25:
            split_horizontally = False
        elif self.height > self.width and self.height / self.width >= 1.25:
            split_horizontally = True

        max_size = (self.height if split_horizontally else self.width) - min_leaf_size
        if max_size <= min_leaf_size:
            return False

        split_point = random.randint(min_leaf_size, max_size)

        if split_horizontally:
            self.child_1 = Leaf(self.x, self.y, self.width, split_point)
            self.child_2 = Leaf(self.x, self.y + split_point, self.width, self.height - split_point)
        else:
            self.child_1 = Leaf(self.x, self.y, split_point, self.height)
            self.child_2 = Leaf(self.x + split_point, self.y, self.width - split_point, self.height)
            
        return True
    
    def create_room(self, room_min_padding):
        # MODIFICADO: room_min_padding es un argumento. Esta es tu lógica mejorada.
        if self.child_1 is not None or self.child_2 is not None:
            return

        # Calcular dimensiones y posición de la habitación de forma segura
        try:
            room_width = random.randint(self.width // 2, self.width - room_min_padding * 2)
            room_height = random.randint(self.height // 2, self.height - room_min_padding * 2)
            room_x = random.randint(self.x + room_min_padding, self.x + self.width - room_width - room_min_padding)
            room_y = random.randint(self.y + room_min_padding, self.y + self.height - room_height - room_min_padding)
            self.room = {'x': room_x, 'y': room_y, 'width': room_width, 'height': room_height}
        except ValueError:
            # Si random.randint falla porque el mínimo es mayor que el máximo, simplemente no crea la habitación.
            self.room = None


def connect_rooms(room1, room2, grid):
    """Talla un pasillo en forma de L entre los centros de dos habitaciones."""
    center1_x = room1['x'] + room1['width'] // 2
    center1_y = room1['y'] + room1['height'] // 2
    center2_x = room2['x'] + room2['width'] // 2
    center2_y = room2['y'] + room2['height'] // 2

    if random.choice([True, False]):
        # Horizontal, luego Vertical
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1):
            grid[center1_y, x] = 1 # 1 representa el suelo
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1):
            grid[y, center2_x] = 1
    else:
        # Vertical, luego Horizontal
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1):
            grid[y, center1_x] = 1
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1):
            grid[center2_y, x] = 1

# --- Función Principal (Wrapper para FastAPI) ---

def create_dungeon_layout(config: dict):
    """
    Función principal que toma una configuración y devuelve un mapa de calabozo.
    """
    # 1. Extraer parámetros de la configuración, con valores por defecto
    MAP_WIDTH = config.get('width', 50)
    MAP_HEIGHT = config.get('height', 30)
    MIN_LEAF_SIZE = config.get('min_leaf_size', 6)
    ROOM_MIN_PADDING = config.get('padding', 1)

    # 2. Partición
    root_leaf = Leaf(0, 0, MAP_WIDTH, MAP_HEIGHT)
    leaves_list = [root_leaf]
    did_split = True
    while did_split:
        did_split = False
        for leaf in list(leaves_list):
            if leaf.child_1 is None and leaf.child_2 is None:
                if leaf.width > MIN_LEAF_SIZE * 2 or leaf.height > MIN_LEAF_SIZE * 2:
                    # Pasar MIN_LEAF_SIZE como argumento
                    if leaf.split(MIN_LEAF_SIZE):
                        leaves_list.append(leaf.child_1)
                        leaves_list.append(leaf.child_2)
                        did_split = True

    # 3. Creación de habitaciones
    final_leaves = [leaf for leaf in leaves_list if leaf.child_1 is None and leaf.child_2 is None]
    for leaf in final_leaves:
        # Pasar ROOM_MIN_PADDING como argumento
        leaf.create_room(ROOM_MIN_PADDING)

    # 4. Conexión de habitaciones
    dungeon_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
    for leaf in final_leaves:
        if leaf.room:
            room = leaf.room
            dungeon_grid[room['y']:room['y']+room['height'], room['x']:room['x']+room['width']] = 1
    
    parent_leaves = [leaf for leaf in leaves_list if leaf.child_1 is not None and leaf.child_2 is not None]
    for parent in parent_leaves:
        if parent.child_1 and parent.child_1.room and parent.child_2 and parent.child_2.room:
            connect_rooms(parent.child_1.room, parent.child_2.room, dungeon_grid)
    
    # 5. Devolver el resultado final
    return dungeon_grid