# Módulo para centralizar as fórmulas de Álgebra Linear e Computação Gráfica
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FÓRMULAS DE CURVAS
# ==============================================================================

def matriz_base_hermite():
    """Retorna a matriz base de Hermite."""
    return np.array([
        [ 2, -2,  1,  1],
        [-3,  3, -2, -1],
        [ 0,  0,  1,  0],
        [ 1,  0,  0,  0]
    ])
    
def hermite(p1, p2, t1, t2, num_pontos=100):
    """
    Gera pontos para uma curva de Hermite, seguindo a fórmula p(t) = T * C, onde C = H * G.
    
    Args:
        p1 (np.array): Ponto inicial.
        p2 (np.array): Ponto final.
        t1 (np.array): Vetor tangente inicial.
        t2 (np.array): Vetor tangente final.
        num_pontos (int): Número de pontos a serem gerados na curva.

    Returns:
        np.array: Array de pontos que formam a curva.
    """
    # T = [t^3, t^2, t, 1]
    t_vetor = np.linspace(0, 1, num_pontos)[:, np.newaxis]
    T = np.hstack([t_vetor**3, t_vetor**2, t_vetor, np.ones_like(t_vetor)])
    
    # G = Matriz de Geometria (pontos de controle e tangentes)
    G = np.vstack([p1, p2, t1, t2])
    
    # H = Matriz Base de Hermite
    H = matriz_base_hermite()
    
    # C = H * G
    # Calcula a matriz de coeficientes do polinômio cúbico.
    C = H @ G
    
    # p(t) = T * C
    # Avalia o polinômio para todos os pontos 't' para obter as coordenadas da curva.
    pontos_da_curva = T @ C
    
    return pontos_da_curva


def derivada_hermite(p1, p2, t1, t2, t):
    """
    Calcula o vetor tangente (derivada) em um ponto t de uma curva de Hermite.
    A fórmula é P'(t) = T' * C, onde T' = [3t^2, 2t, 1, 0] e C = H * G.
    """
    # T' = Vetor de tempo derivado
    T_deriv = np.array([3*t**2, 2*t, 1, 0])
    
    # G = Matriz de Geometria
    G = np.vstack([p1, p2, t1, t2])
    
    # H = Matriz Base de Hermite
    H = matriz_base_hermite()
    
    # C = H * G (Coeficientes do polinômio)
    C = H @ G
    
    # P'(t) = T' * C (Tangente)
    tangente = T_deriv @ C
    
    return tangente



# ==============================================================================
# MATRIZES DE TRANSFORMAÇÃO GEOMÉTRICA (4x4 para 3D)
# ==============================================================================

def matriz_translacao(tx, ty, tz):
    """
    Retorna uma matriz de translação 4x4.
    Equivalente 3D da sua matriz [1 0 tx; 0 1 ty; 0 0 1].
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ])

def matriz_escala(sx, sy, sz):
    """
    Retorna uma matriz de escala 4x4.
    Equivalente 3D da sua matriz [sx 0 0; 0 sy 0; 0 0 1].
    """
    return np.array([
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1]
    ])

# --- Rotações Individuais nos Eixos ---

def matriz_rotacao_z(angulo_rad):
    """
    Retorna uma matriz de rotação 4x4 em torno do eixo Z.
    Esta é a rotação 3D análoga à sua matriz de rotação 2D.
    """
    c = np.cos(angulo_rad)
    s = np.sin(angulo_rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

def matriz_rotacao_y(angulo_rad):
    """Retorna uma matriz de rotação 4x4 em torno do eixo Y."""
    c = np.cos(angulo_rad)
    s = np.sin(angulo_rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

def matriz_rotacao_x(angulo_rad):
    """Retorna uma matriz de rotação 4x4 em torno do eixo X."""
    c = np.cos(angulo_rad)
    s = np.sin(angulo_rad)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ])
    
# ==============================================================================
# FÓRMULAS DE CÂMERA (VIEW MATRIX)
# ==============================================================================

def matriz_visao(eye, at, aux=np.array([0, 1, 0])):
    """
    Calcula a matriz de visão (View Matrix) 4x4.
    A fórmula final é ViewMatrix = R @ T.
    """
    # 1. Calcular a base vetorial da câmera (n, u, v) - Lógica inalterada
    n = normalizar_vetor(at - eye)
    v = normalizar_vetor(np.cross(n, aux)) # Temporário para calcular u
    u = normalizar_vetor(np.cross(v, n))
    # Recalcula v para garantir a ortogonalidade, caso 'aux' não seja perfeitamente perpendicular
    # v = np.cross(n, u)

    # 2. Construir a Matriz de Translação (T) - Lógica inalterada
    T = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])

    # 3. CONSTRUIR A MATRIZ DE ROTAÇÃO (R) COM A ORDEM N, U, V
    # AJUSTE FEITO AQUI: A ordem das linhas foi alterada para N, U, V.
    R = np.array([
        [n[0], n[1], n[2], 0], # Eixo N (direção da visão)
        [u[0], u[1], u[2], 0], # Eixo U (para cima)
        [v[0], v[1], v[2], 0], # Eixo V (para a direita)
        [0,    0,    0,    1]
    ])

    # 4. Retornar a Matriz de Visão final (View = R * T) - Lógica inalterada
    return R @ T

def matriz_projecao_perspectiva(fov_graus, aspect_ratio, near, far):
    """
    Calcula a matriz de projeção 4x4 customizada para ser compatível com uma
    matriz de visão que retorna coordenadas na ordem (N, U, V).
    """
    fov_rad = np.radians(fov_graus)
    t = np.tan(fov_rad / 2) # Fator de escala para a altura (U)
    
    # Coeficientes para mapear a profundidade (N) para o intervalo [0, 1]
    A = far / (far - near)
    B = - (far * near) / (far - near)
    
    # Monta a matriz que remapeia as componentes de N, U, V para X, Y, Z, W
    return np.array([
        # Coluna de saída:  X (tela)                 Y (tela)                  Z (buffer)            W (divisão)
        # ------------------------------------------------------------------------------------------------------------------
        [0,                       0,                1 / (aspect_ratio * t),        0],  # Mapeia V para a saída X
        [0,                       1 / t,            0,                             0],  # Mapeia U para a saída Y
        [A,                       0,                0,                             B],  # Mapeia N para a saída Z
        [1,                       0,                0,                             0]   # Mapeia N para a saída W
    ])

# ==============================================================================
# FUNÇÕES AUXILIARES DE ÁLGEBRA LINEAR
# ==============================================================================

def normalizar_vetor(v):
    """Normaliza um vetor para ter magnitude 1."""
    norma = np.linalg.norm(v)
    if norma == 0:
        return v
    return v / norma

def transformar_pontos(pontos, matriz_transformacao):
    """
    Aplica uma matriz de transformação 4x4 a um conjunto de pontos.
    """
    # Adiciona a coordenada homogênea (w=1)
    pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
    
    # Aplica a transformação
    pontos_transformados_homogeneos = (matriz_transformacao @ pontos_homogeneos.T).T
    
    # Converte de volta para 3D (se necessário)
    # A divisão por w é feita na etapa de projeção
    return pontos_transformados_homogeneos[:, :3]



# ==============================================================================
# ALGORITMOS DE RASTERIZAÇÃO 2D
# ==============================================================================

def rasterizar_linha_bresenham(p1, p2):
    """Rasteriza uma linha 2D usando o Algoritmo de Bresenham."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    pixels = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    trocado = dy > dx
    if trocado:
        dx, dy = dy, dx
    p = 2 * dy - dx
    x, y = x1, y1
    for _ in range(dx + 1):
        pixels.append((x, y))
        if p >= 0:
            if trocado: x += sx
            else: y += sy
            p -= 2 * dx
        if trocado: y += sy
        else: x += sx
        p += 2 * dy
    return pixels

def rasterizar_poligono_scanline(vertices):
    """Rasteriza um polígono 2D usando o Algoritmo Scan-Line."""
    if len(vertices) < 3: return []
    pixels = []
    min_y = int(np.ceil(min(v[1] for v in vertices)))
    max_y = int(np.floor(max(v[1] for v in vertices)))
    edge_table = {y: [] for y in range(min_y, max_y + 1)}
    num_vertices = len(vertices)
    for i in range(num_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % num_vertices]
        if p1[1] > p2[1]: p1, p2 = p2, p1
        y_min_aresta, y_max_aresta = p1[1], p2[1]
        if y_min_aresta == y_max_aresta: continue
        x_na_y_min = p1[0]
        inclinacao_inversa = (p2[0] - p1[0]) / (p2[1] - p1[1])
        edge_table[int(np.ceil(y_min_aresta))].append([y_max_aresta, x_na_y_min, inclinacao_inversa])
    
    active_edge_table = []
    for y in range(min_y, max_y + 1):
        for aresta in edge_table[y]:
            active_edge_table.append(aresta)
        active_edge_table = [aresta for aresta in active_edge_table if aresta[0] > y]
        active_edge_table.sort(key=lambda aresta: aresta[1])
        for i in range(0, len(active_edge_table), 2):
            if i + 1 < len(active_edge_table):
                x_inicio = int(np.ceil(active_edge_table[i][1]))
                x_fim = int(np.floor(active_edge_table[i+1][1]))
                for x in range(x_inicio, x_fim + 1):
                    pixels.append((x, y))
        for aresta in active_edge_table:
            aresta[1] += aresta[2]
    return pixels

