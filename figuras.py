import numpy as np
import random

# base
class FiguraBase:
    def __init__(self, nombre):
        self.nombre = nombre

    def tallar(self, grid, *args, **kwargs):
        raise NotImplementedError("Este método debe implementarse en la subclase.")


#figuras
class Circulo(FiguraBase):
    def __init__(self):
        super().__init__("circulo")

    def tallar(self, grid, x, y, radio):
        for i in range(x - radio, x + radio):
            for j in range(y - radio, y + radio):
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                    if (i - x) ** 2 + (j - y) ** 2 <= radio ** 2:
                        grid[i, j] = 1
        return grid


class Elipse(FiguraBase):
    def __init__(self):
        super().__init__("elipse")

    def tallar(self, grid, x, y, rx, ry):
        for i in range(x - rx, x + rx):
            for j in range(y - ry, y + ry):
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                    if ((i - x) ** 2) / (rx ** 2) + ((j - y) ** 2) / (ry ** 2) <= 1:
                        grid[i, j] = 1
        return grid


class Rectangulo(FiguraBase):
    def __init__(self):
        super().__init__("rectangulo")

    def tallar(self, grid, x, y, ancho, alto):
        for i in range(x, min(x + ancho, grid.shape[0])):
            for j in range(y, min(y + alto, grid.shape[1])):
                grid[i, j] = 1
        return grid


class Triangulo(FiguraBase):
    def __init__(self):
        super().__init__("triangulo")

    def tallar(self, grid, x, y, base, altura):
        for i in range(altura):
            for j in range(base - i * base // altura):
                xi = x + i
                yj = y + j + (i // 2)
                if 0 <= xi < grid.shape[0] and 0 <= yj < grid.shape[1]:
                    grid[xi, yj] = 1
        return grid


class CaminataAleatoria(FiguraBase):
    def __init__(self):
        super().__init__("caminata_aleatoria")

    def tallar(self, grid, x, y, pasos):
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        for _ in range(pasos):
            grid[x, y] = 1
            dir = random.randint(0, 3)
            x += dx[dir]
            y += dy[dir]
            x = max(0, min(x, grid.shape[0] - 1))
            y = max(0, min(y, grid.shape[1] - 1))
        return grid



# gestor
class GestorFiguras:
    def __init__(self, limite=5):
        self.limite = limite
        self.registro = []
        self.catalogo = {
            "circulo": Circulo(),
            "elipse": Elipse(),
            "rectangulo": Rectangulo(),
            "triangulo": Triangulo(),
            "caminata_aleatoria": CaminataAleatoria()
        }

    def agregar_figura(self, nombre):
        """Agrega una figura si no se ha superado el límite."""
        if len(self.registro) >= self.limite:
            print(f"Límite de {self.limite} figuras alcanzado. No se agregará '{nombre}'.")
            return False
        if nombre not in self.catalogo:
            print(f"Figura '{nombre}' no reconocida.")
            return False
        self.registro.append(self.catalogo[nombre])
        print(f"Figura '{nombre}' agregada.")
        return True

    def generar(self, ancho=100, alto=100):
        """Genera el mapa con las figuras seleccionadas."""
        grid = np.zeros((ancho, alto))
        for figura in self.registro:
            # Coordenadas aleatorias
            x = random.randint(10, ancho - 10)
            y = random.randint(10, alto - 10)

            # Parámetros
            if figura.nombre == "circulo":
                grid = figura.tallar(grid, x, y, random.randint(5, 10))
            elif figura.nombre == "elipse":
                grid = figura.tallar(grid, x, y, random.randint(5, 12), random.randint(3, 8))
            elif figura.nombre == "rectangulo":
                grid = figura.tallar(grid, x, y, random.randint(8, 15), random.randint(6, 10))
            elif figura.nombre == "triangulo":
                grid = figura.tallar(grid, x, y, random.randint(8, 15), random.randint(6, 10))
            elif figura.nombre == "caminata_aleatoria":
                grid = figura.tallar(grid, x, y, random.randint(100, 300))

        return grid

    def listar_figuras_disponibles(self):
        return list(self.catalogo.keys())

    def limpiar(self):
        self.registro.clear()
        print("🧹 Registro de figuras limpiado.")



# prueba para el test local
if __name__ == "__main__":
    gestor = GestorFiguras(limite=3)
    print("Figuras disponibles:", gestor.listar_figuras_disponibles())

    gestor.agregar_figura("circulo")
    gestor.agregar_figura("rectangulo")
    gestor.agregar_figura("caminata_aleatoria")

    mapa = gestor.generar(60, 60)
    print("Mapa generado con las figuras seleccionadas.")
    print(mapa)
