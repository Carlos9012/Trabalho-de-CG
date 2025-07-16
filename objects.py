import numpy as np
import math

class Cano:
    def __init__(self, raio_externo, comprimento, espessura=0.1, n_segmentos=32, n_cortes=20):
        self.vertices, self.faces = self._gerar_cano(raio_externo, comprimento, espessura, n_segmentos, n_cortes)

    @staticmethod
    def _gerar_cano(raio_externo, comprimento, espessura, n_segmentos, n_cortes):
        raio_interno = raio_externo - espessura
        vertices = []
        faces = []

        # Gerar vértices para cada segmento ao longo do comprimento
        for k in range(n_cortes + 1):
            z = comprimento * k / n_cortes
            for i in range(n_segmentos):
                theta = 2 * math.pi * i / n_segmentos
                
                # Vértice externo
                x_ext = raio_externo * math.cos(theta)
                y_ext = raio_externo * math.sin(theta)
                vertices.append([x_ext, y_ext, z])
                
                # Vértice interno
                x_int = raio_interno * math.cos(theta)
                y_int = raio_interno * math.sin(theta)
                vertices.append([x_int, y_int, z])

        # Gerar faces para a superfície lateral
        for k in range(n_cortes):
            for i in range(n_segmentos):
                next_i = (i + 1) % n_segmentos
                
                # Índices dos vértices
                a_ext = 2 * (k * n_segmentos + i)
                b_ext = 2 * (k * n_segmentos + next_i)
                a_int = a_ext + 1
                b_int = b_ext + 1
                
                a_next_ext = 2 * ((k + 1) * n_segmentos + i)
                b_next_ext = 2 * ((k + 1) * n_segmentos + next_i)
                a_next_int = a_next_ext + 1
                b_next_int = b_next_ext + 1

                # Face externa
                faces.append([a_ext, a_next_ext, b_next_ext])
                faces.append([a_ext, b_next_ext, b_ext])
                
                # Face interna
                faces.append([a_int, b_next_int, a_next_int])
                faces.append([a_int, b_int, b_next_int])
                
                # Face superior (conectando interno e externo)
                if k == 0:  # Primeira extremidade
                    faces.append([a_ext, b_ext, a_int])
                    faces.append([b_ext, b_int, a_int])
                
                # Face inferior (conectando interno e externo)
                if k == n_cortes - 1:  # Última extremidade
                    faces.append([a_next_ext, a_next_int, b_next_ext])
                    faces.append([b_next_ext, a_next_int, b_next_int])

        return vertices, faces


class CanoCurvo:
    def __init__(self, raio_externo=1.0, comprimento=5.0, espessura=0.2, n_segmentos=32, pontos_controle=None):
        self.raio_externo = raio_externo
        self.comprimento = comprimento
        self.espessura = espessura
        self.n_segmentos = n_segmentos

        self.pontos_controle = np.array(pontos_controle or [
            [0, 0, 0],
            [comprimento / 3, comprimento / 3, comprimento / 3],
            [2 * comprimento / 3, -comprimento / 3, 2 * comprimento / 3],
            [comprimento, 0, comprimento]
        ])

        self.vertices, self.faces = self._gerar_malha()

    def _bezier(self, t):
        return (1 - t) ** 3 * self.pontos_controle[0] + \
               3 * (1 - t) ** 2 * t * self.pontos_controle[1] + \
               3 * (1 - t) * t ** 2 * self.pontos_controle[2] + \
               t ** 3 * self.pontos_controle[3]

    def _derivada_bezier(self, t):
        return 3 * (1 - t) ** 2 * (self.pontos_controle[1] - self.pontos_controle[0]) + \
               6 * (1 - t) * t * (self.pontos_controle[2] - self.pontos_controle[1]) + \
               3 * t ** 2 * (self.pontos_controle[3] - self.pontos_controle[2])

    def _gerar_malha(self):
        vertices = []
        faces = []

        t_vals = np.linspace(0, 1, self.n_segmentos)
        pontos = np.array([self._bezier(t) for t in t_vals])
        tangentes = np.array([self._derivada_bezier(t) for t in t_vals])

        for ponto, tangente in zip(pontos, tangentes):
            tangente_norm = tangente / np.linalg.norm(tangente)
            aux = np.array([0, 0, 1]) if abs(tangente_norm[0]) > 0.1 or abs(tangente_norm[1]) > 0.1 else np.array([1, 0, 0])
            normal = np.cross(tangente_norm, aux)
            normal = normal / np.linalg.norm(normal)
            binormal = np.cross(tangente_norm, normal)

            for j in range(self.n_segmentos):
                theta = 2 * np.pi * j / self.n_segmentos
                dir_v = normal * np.cos(theta) + binormal * np.sin(theta)
                vertices.append(ponto + dir_v * self.raio_externo)
                vertices.append(ponto + dir_v * (self.raio_externo - self.espessura))

        for i in range(len(t_vals) - 1):
            for j in range(self.n_segmentos):
                next_j = (j + 1) % self.n_segmentos
                a0 = i * 2 * self.n_segmentos + 2 * j
                a1 = a0 + 1
                b0 = (i + 1) * 2 * self.n_segmentos + 2 * j
                b1 = b0 + 1
                c0 = (i + 1) * 2 * self.n_segmentos + 2 * next_j
                c1 = c0 + 1
                d0 = i * 2 * self.n_segmentos + 2 * next_j
                d1 = d0 + 1

                faces.extend([[a0, b0, c0], [a0, c0, d0], [a1, c1, b1], [a1, d1, c1]])
                if i == 0:
                    faces.extend([[a0, d0, a1], [a1, d0, d1]])
                if i == len(t_vals) - 2:
                    faces.extend([[b0, b1, c0], [c0, b1, c1]])

        return np.array(vertices), np.array(faces)


class Cilindro:
    def __init__(self, raio, altura, n=32, m=20):
        self.vertices, self.faces = self._gerar_malha(raio, altura, n, m)

    @staticmethod
    def _gerar_malha(raio, altura, n, m):
        vertices = []

        for k in range(m + 1):
            z = altura * k / m
            for i in range(n):
                theta = 2 * math.pi * i / n
                x = raio * math.cos(theta)
                y = raio * math.sin(theta)
                vertices.append([x, y, z])

        
        idx_base_center = len(vertices)
        vertices.append([0, 0, 0])         
        idx_top_center = len(vertices)
        vertices.append([0, 0, altura])     

        faces = []
        
        for k in range(m):
            offset0 = k * n
            offset1 = (k + 1) * n
            for i in range(n):
                next_i = (i + 1) % n
                a, b = offset0 + i, offset1 + i
                c, d = offset0 + next_i, offset1 + next_i
                faces.extend([[a, b, d], [a, d, c]])

        
        for i in range(n):
            next_i = (i + 1) % n
            
            faces.append([idx_base_center, next_i, i])
            
            faces.append([idx_top_center, (m * n) + i, (m * n) + next_i])

        return np.array(vertices), np.array(faces)


class LinhaReta4:
    def __init__(self, comprimento=4.0, espessura=0.1):
        self.vertices, self.faces = self._gerar_malha(comprimento, espessura)

    @staticmethod
    def _gerar_malha(comp, e):
        v = np.array([
            [-e/2, -e/2, 0],  [ e/2, -e/2, 0],
            [ e/2,  e/2, 0],  [-e/2,  e/2, 0],
            [-e/2, -e/2, comp], [ e/2, -e/2, comp],
            [ e/2,  e/2, comp], [-e/2,  e/2, comp]
        ])

        f = np.array([
            [0,1,2], [0,2,3],          # base
            [4,6,5], [4,7,6],          # topo
            [0,4,1], [1,4,5],          # lado 1
            [1,5,2], [2,5,6],          # lado 2
            [2,6,3], [3,6,7],          # lado 3
            [3,7,0], [0,7,4]           # lado 4
        ])

        return v, f
    
    
class Paralelepipedo:
    def __init__(self, largura, altura, espessura, n=1, m=1, l=1):
        self.vertices, self.faces = self._gerar_malha(largura, altura, espessura, n, m, l)

    @staticmethod
    def _gerar_malha(largura, altura, espessura, n, m, l):
        # Divisões ao longo de X, Y, Z
        dx = largura / n
        dy = espessura / l
        dz = altura / m

        vertices = []
        index_map = {}

        # Gera os vértices na grade 3D
        idx = 0
        for k in range(m + 1):       # Altura (Z)
            z = dz * k
            for j in range(l + 1):   # Espessura (Y)
                y = -espessura / 2 + dy * j
                for i in range(n + 1):  # Largura (X)
                    x = -largura / 2 + dx * i
                    vertices.append([x, y, z])
                    index_map[(i, j, k)] = idx
                    idx += 1

        faces = []

        # Faces laterais ao longo da altura (m), subdivididas em X (n)
        for j in range(l):  # frente e trás
            for i in range(n):
                for k in range(m):
                    # Face frontal
                    a = index_map[(i, j, k)]
                    b = index_map[(i + 1, j, k)]
                    c = index_map[(i + 1, j, k + 1)]
                    d = index_map[(i, j, k + 1)]
                    faces.extend([[a, b, c], [a, c, d]])

                    # Face traseira
                    a = index_map[(i, j + 1, k)]
                    b = index_map[(i + 1, j + 1, k)]
                    c = index_map[(i + 1, j + 1, k + 1)]
                    d = index_map[(i, j + 1, k + 1)]
                    faces.extend([[d, c, b], [d, b, a]])

        # Faces laterais direita e esquerda
        for i in range(n + 1):  # fixa em i
            for j in range(l):
                for k in range(m):
                    a = index_map[(i, j, k)]
                    b = index_map[(i, j + 1, k)]
                    c = index_map[(i, j + 1, k + 1)]
                    d = index_map[(i, j, k + 1)]
                    faces.extend([[a, b, c], [a, c, d]])

        # Faces superior e inferior
        for k in [0, m]:  # z fixo (baixo e topo)
            for j in range(l):
                for i in range(n):
                    a = index_map[(i, j, k)]
                    b = index_map[(i + 1, j, k)]
                    c = index_map[(i + 1, j + 1, k)]
                    d = index_map[(i, j + 1, k)]
                    if k == 0:  # inferior
                        faces.extend([[a, b, c], [a, c, d]])
                    else:  # superior
                        faces.extend([[d, c, b], [d, b, a]])

        return np.array(vertices), np.array(faces)
