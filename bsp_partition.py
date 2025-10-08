import random
import numpy as np
import math

# --- Constantes ---
MAP_WIDTH = 50
MAP_HEIGHT = 30
MIN_LEAF_SIZE = 6
ROOM_MIN_SIZE = 4

class Leaf:
    # __init__ y split() no cambian
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.child_1 = None
        self.child_2 = None
        self.room = None

    def split(self):
        if self.child_1 is not None or self.child_2 is not None: return False
        split_horizontally = random.choice([True, False])
        if self.width > self.height and self.width / self.height >= 1.25: split_horizontally = False
        elif self.height > self.width and self.height / self.width >= 1.25: split_horizontally = True
        max_size = (self.height if split_horizontally else self.width) - MIN_LEAF_SIZE
        if max_size <= MIN_LEAF_SIZE: return False
        split_point = random.randint(MIN_LEAF_SIZE, max_size)
        if split_horizontally:
            self.child_1 = Leaf(self.x, self.y, self.width, split_point)
            self.child_2 = Leaf(self.x, self.y + split_point, self.width, self.height - split_point)
        else:
            self.child_1 = Leaf(self.x, self.y, split_point, self.height)
            self.child_2 = Leaf(self.x + split_point, self.y, self.width - split_point, self.height)
        return True
    
    # --- CAMBIO PRINCIPAL: Lógica de Sesgo de Forma ---
    def create_room(self):
        if self.child_1 is not None or self.child_2 is not None:
            return

        padding = 2
        # Comprobación de seguridad: ¿Hay espacio suficiente para la habitación más pequeña?
        if self.width < ROOM_MIN_SIZE + padding or self.height < ROOM_MIN_SIZE + padding:
            self.room = None
            return

        # Decidir aleatoriamente el sesgo de la forma de la habitación
        shape_bias = random.choice(['wide', 'tall', 'normal'])
        
        try:
            if shape_bias == 'wide':
                # Forzar una habitación ancha
                min_w = int(self.width * 0.7) # Mínimo 70% del ancho disponible
                max_w = self.width - padding
                min_h = ROOM_MIN_SIZE
                max_h = int(self.height * 0.6) # Máximo 60% del alto disponible
            elif shape_bias == 'tall':
                # Forzar una habitación alta
                min_w = ROOM_MIN_SIZE
                max_w = int(self.width * 0.6)
                min_h = int(self.height * 0.7)
                max_h = self.height - padding
            else: # 'normal'
                # Comportamiento equilibrado
                min_w = ROOM_MIN_SIZE
                max_w = self.width - padding
                min_h = ROOM_MIN_SIZE
                max_h = self.height - padding

            # Asegurarse de que los mínimos no superen a los máximos
            actual_min_w = max(ROOM_MIN_SIZE, min_w)
            actual_max_w = max(actual_min_w, max_w)
            actual_min_h = max(ROOM_MIN_SIZE, min_h)
            actual_max_h = max(actual_min_h, max_h)

            room_width = random.randint(actual_min_w, actual_max_w)
            room_height = random.randint(actual_min_h, actual_max_h)

            # La posición también es más variable
            room_x = random.randint(self.x + 1, self.x + self.width - room_width - 1)
            room_y = random.randint(self.y + 1, self.y + self.height - room_height - 1)

            self.room = {'x': room_x, 'y': room_y, 'width': room_width, 'height': room_height}
        except ValueError:
            # Si algo falla en los cálculos, simplemente no se crea la habitación
            self.room = None

# El resto de funciones (connect_rooms, get_room_from_branch) y la lógica principal no cambian
def connect_rooms(room1, room2, grid):
    center1_x = room1['x'] + room1['width'] // 2
    center1_y = room1['y'] + room1['height'] // 2
    center2_x = room2['x'] + room2['width'] // 2
    center2_y = room2['y'] + room2['height'] // 2
    if random.choice([True, False]):
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1): grid[center1_y, x] = 1
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1): grid[y, center2_x] = 1
    else:
        for y in range(min(center1_y, center2_y), max(center1_y, center2_y) + 1): grid[y, center1_x] = 1
        for x in range(min(center1_x, center2_x), max(center1_x, center2_x) + 1): grid[center2_y, x] = 1

def get_room_from_branch(leaf):
    if leaf.room: return leaf.room
    else:
        room1, room2 = None, None
        if leaf.child_1: room1 = get_room_from_branch(leaf.child_1)
        if leaf.child_2: room2 = get_room_from_branch(leaf.child_2)
        if room1 is None and room2 is None: return None
        elif room1 is None: return room2
        elif room2 is None: return room1
        else: return random.choice([room1, room2])

# --- Lógica principal ---
root_leaf = Leaf(0, 0, MAP_WIDTH, MAP_HEIGHT)
leaves_list = [root_leaf]
did_split = True
while did_split:
    did_split = False
    for leaf in list(leaves_list):
        if leaf.child_1 is None and leaf.child_2 is None:
            if leaf.width > MIN_LEAF_SIZE * 2 or leaf.height > MIN_LEAF_SIZE * 2:
                if leaf.split():
                    leaves_list.append(leaf.child_1)
                    leaves_list.append(leaf.child_2)
                    did_split = True

final_leaves = [leaf for leaf in leaves_list if leaf.child_1 is None and leaf.child_2 is None]
for leaf in final_leaves:
    leaf.create_room()

dungeon_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
for leaf in final_leaves:
    if leaf.room:
        room = leaf.room
        dungeon_grid[room['y']:room['y']+room['height'], room['x']:room['x']+room['width']] = 1

parent_leaves = [leaf for leaf in leaves_list if leaf.child_1 is not None and leaf.child_2 is not None]
for parent in parent_leaves:
    room1 = get_room_from_branch(parent.child_1)
    room2 = get_room_from_branch(parent.child_2)
    if room1 and room2:
        connect_rooms(room1, room2, dungeon_grid)

# --- Resultado ---
print("Mapa Lógico Generado con Sesgo de Forma:")
for y in range(MAP_HEIGHT):
    line = ""
    for x in range(MAP_WIDTH):
        line += "#" if dungeon_grid[y, x] == 0 else "."
    print(line)