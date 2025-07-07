import numpy as np
import math

class Cano:
    def __init__(self, raio_externo, comprimento, espessura=0.1, n=32, m=20):
        self.vertices, self.faces = self._gerar_malha(raio_externo, comprimento, espessura, n, m)

    @staticmethod
    def _gerar_malha(raio_externo, comprimento, espessura=0.1, n=32, m=20):
        raio_interno = raio_externo - espessura
        vertices = []

        for k in range(m + 1):
            z = comprimento * k / m
            for r in [raio_externo, raio_interno]:
                for i in range(n):
                    theta = 2 * math.pi * i / n
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    vertices.append([x, y, z])

        faces = []
        for k in range(m):
            offset0 = k * 2 * n
            offset1 = (k + 1) * 2 * n
            for i in range(n):
                next_i = (i + 1) % n

                
                a, b = offset0 + i, offset1 + i
                c, d = offset0 + next_i, offset1 + next_i
                faces.extend([[a, b, d], [a, d, c]])

                a_int, b_int = offset0 + n + i, offset1 + n + i
                c_int, d_int = offset0 + n + next_i, offset1 + n + next_i
                faces.extend([[b_int, a_int, c_int], [b_int, c_int, d_int]])

        
        for i in range(n):
            next_i = (i + 1) % n
            faces.extend([
                [i + n, next_i + n, i],               
                [next_i + n, next_i, i],
                [m * 2 * n + i, m * 2 * n + next_i, m * 2 * n + i + n],  
                [m * 2 * n + next_i, m * 2 * n + next_i + n, m * 2 * n + i + n]
            ])

        return np.array(vertices), np.array(faces)



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