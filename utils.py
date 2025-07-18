import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch 
from objects import Cano, CanoCurvo, Cilindro, LinhaReta, Paralelepipedo


def _transformar_vertices(vertices, escala=1.0, rotacao=np.eye(3), translacao=np.zeros(3)):
    S = np.diag([escala, escala, escala])
    return (vertices @ S @ rotacao.T) + translacao

def plotar(objeto, mostrar_malha=True, cor='skyblue', title='Figura'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    v = np.array(objeto.vertices)
    f = np.array(objeto.faces)
    
    # Verifica se é uma linha (sem faces) ou um objeto com malha
    if len(f) > 0:
        # Plot para objetos com malha 3D
        mesh = Poly3DCollection(
            v[f],
            alpha=0.6,
            linewidths=0.3 if mostrar_malha else 0,
            edgecolor='k' if mostrar_malha else 'none',
            facecolor=cor
        )
        ax.add_collection3d(mesh)
    else:
        # Plot para linhas (apenas vértices)
        if len(v) == 2:
            # Linha simples (dois pontos)
            ax.plot(v[:,0], v[:,1], v[:,2], 
                   color=cor, 
                   linewidth=2, 
                   marker='o',
                   markersize=6)
        else:
            # Linha poligonal (múltiplos pontos)
            ax.plot(v[:,0], v[:,1], v[:,2], 
                   color=cor, 
                   linewidth=2)

    # Ajuste dos limites do gráfico
    todos_v = np.vstack(v)
    max_c = np.max(np.abs(todos_v)) * 1.1
    ax.set_xlim(-max_c, max_c)
    ax.set_ylim(-max_c, max_c)
    ax.set_zlim(-max_c, max_c)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def compor_cena_varios(objetos_transformados: list[tuple],
                      mostrar_malha: bool = True,
                      title: str = "Cena 3D com múltiplos objetos",
                      tam_max: float = 10.0):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    tamanhos_maximos = []
    offset = 0
    todos_vertices = []
    legend_elements = []

    for i, (obj, esc, rot, trans, cor) in enumerate(objetos_transformados):
        vertices_originais = np.array(obj.vertices)

        # Transformar vértices
        v = _transformar_vertices(vertices_originais, esc, rot, trans)
        max_extent = np.max(np.linalg.norm(v, axis=1)) * 1.2

        # Reduzir escala se exceder tam_max
        if max_extent > tam_max:
            fator_escala = tam_max / max_extent
            esc = esc * fator_escala
            v = _transformar_vertices(vertices_originais, esc, rot, trans)
            max_extent = np.max(np.linalg.norm(v, axis=1)) * 1.2

        tamanhos_maximos.append((i, max_extent))
        todos_vertices.append(v)

        # Verificar se é objeto 3D (com faces) ou linha (sem faces)
        if len(obj.faces) > 0:
            # Objeto 3D com malha
            faces_validas = []
            for face in obj.faces:
                if all(idx < len(v) for idx in face):
                    faces_validas.append(face)

            f = np.array(faces_validas)
            
            mesh = Poly3DCollection(
                v[f],
                alpha=0.6,
                linewidths=0.3 if mostrar_malha else 0,
                edgecolor='k' if mostrar_malha else 'none',
                facecolor=cor
            )
            ax.add_collection3d(mesh)
        else:
            # Linha (sem faces)
            if len(v) == 2:
                # Linha simples (dois pontos)
                ax.plot(v[:,0], v[:,1], v[:,2], 
                       color=cor, 
                       linewidth=2, 
                       marker='o',
                       markersize=6)
            else:
                # Linha poligonal (múltiplos pontos)
                ax.plot(v[:,0], v[:,1], v[:,2], 
                       color=cor, 
                       linewidth=2)

        legend_elements.append(Patch(facecolor=cor, label=f'Obj {i+1}'))

    # Ajustar os eixos
    if todos_vertices:
        todos_vertices = np.vstack(todos_vertices)
        centro = np.mean(todos_vertices, axis=0)
        max_dist = np.max(np.linalg.norm(todos_vertices - centro, axis=1)) * 1.5
    else:
        centro = np.array([0, 0, 0])
        max_dist = 10.0

    ax.set_xlim(centro[0] - max_dist, centro[0] + max_dist)
    ax.set_ylim(centro[1] - max_dist, centro[1] + max_dist)
    ax.set_zlim(centro[2] - max_dist, centro[2] + max_dist)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\nEscala máxima: {tam_max}")
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

def mostrar_cena_camera_3D(objetos, eye, at, up=np.array([0, 1, 0]), mostrar_malha=True, mostrar_linha_camera=True):
    # Base da câmera
    n = (eye - at)
    n = n / (np.linalg.norm(n) + 1e-8)

    if np.allclose(np.cross(up, n), 0):
        up = np.array([0, 0, 1]) if not np.allclose(up, [0, 0, 1]) else np.array([1, 0, 0])

    u = np.cross(up, n)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)

    # Matriz de rotação da câmera
    R = np.stack([u, v, n], axis=0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    todos_vertices = []

    # Plotar objetos
    for obj, esc, rot, trans, cor in objetos:
        v_world = _transformar_vertices(np.array(obj.vertices), esc, rot, trans)
        v_camera = np.dot(v_world - eye, R.T)
        todos_vertices.append(v_camera)

        # Verificar se é objeto 3D (com faces) ou linha (sem faces)
        if len(obj.faces) > 0:
            # Objeto 3D com malha
            faces_validas = []
            for face in obj.faces:
                if all(idx < len(v_camera) for idx in face):
                    faces_validas.append(face)

            mesh = Poly3DCollection(
                v_camera[faces_validas],
                alpha=0.6,
                linewidths=0.3 if mostrar_malha else 0,
                edgecolor='k' if mostrar_malha else 'none',
                facecolor=cor
            )
            ax.add_collection3d(mesh)
        else:
            # Linha (sem faces)
            if len(v_camera) == 2:
                # Linha simples (dois pontos)
                ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2], 
                       color=cor, 
                       linewidth=2, 
                       marker='o',
                       markersize=6)
            else:
                # Linha poligonal (múltiplos pontos)
                ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2], 
                       color=cor, 
                       linewidth=2)

    # Calcular limites da cena
    if todos_vertices:
        todos_vertices = np.vstack(todos_vertices)
        centro = np.mean(todos_vertices, axis=0)
        max_dist = np.max(np.linalg.norm(todos_vertices - centro, axis=1)) * 1.5
    else:
        centro = np.array([0, 0, 0])
        max_dist = 10.0

    # Visualização dos eixos e pontos de origem (mesmo código anterior)
    origem_mundo_camera = np.dot(-eye, R.T)
    origem_camera = eye

    if mostrar_linha_camera:
        ax.plot([origem_camera[0], origem_mundo_camera[0]],
                [origem_camera[1], origem_mundo_camera[1]],
                [origem_camera[2], origem_mundo_camera[2]],
                'k--', linewidth=1, alpha=0.5, label='Sistema Mundo→Câmera')

    ax.scatter(*origem_camera, color='blue', s=100, label='Origem Câmera')
    ax.scatter(*origem_mundo_camera, color='red', s=100, label='Origem Mundo (0,0,0)')

    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], 
              R[0,0], R[0,1], R[0,2], color='r', length=max_dist/5, label='u (Right)')
    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], 
              R[1,0], R[1,1], R[1,2], color='g', length=max_dist/5, label='v (Up)')
    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], 
              R[2,0], R[2,1], R[2,2], color='b', length=max_dist/5, label='n (Forward)')

    ax.set_xlim(centro[0]-max_dist, centro[0]+max_dist)
    ax.set_ylim(centro[1]-max_dist, centro[1]+max_dist)
    ax.set_zlim(centro[2]-max_dist, centro[2]+max_dist)

    ax.set_xlabel("u (Right)")
    ax.set_ylabel("v (Up)")
    ax.set_zlabel("n (Forward)")
    ax.set_title("Objetos no Sistema da Câmera")
    ax.legend()

    plt.tight_layout()
    plt.show()

def projetar_perspectiva_2d(objetos, eye, at, up=np.array([0, 1, 0]), fov=60, aspect_ratio=1, near=1, far=100, desenhar_faces=True, alpha=0.6, padding=0.1):
    # Base da câmera (view matrix)
    n = (eye - at)
    n = n / (np.linalg.norm(n) + 1e-8)
    
    if np.allclose(np.cross(up, n), 0):
        up = np.array([0, 0, 1]) if not np.allclose(up, [0, 0, 1]) else np.array([1, 0, 0])
    
    u = np.cross(up, n)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    R = np.stack([u, v, n], axis=0)

    # Matriz de projeção
    alpha_rad = np.radians(fov)
    t = np.tan(alpha_rad / 2)

    A = far / (far - near)
    B = -far * near / (far - near)

    P = np.array([
        [1 / (aspect_ratio * t), 0, 0, 0],
        [0, 1 / t, 0, 0],
        [0, 0, A, B],
        [0, 0, 1, 0]
    ])

    fig, ax = plt.subplots(figsize=(8, 8))
    todos_vertices_2d = []

    for obj, esc, rot, trans, cor in objetos:
        vertices_originais = np.array(obj.vertices)
        v = _transformar_vertices(vertices_originais, esc, rot, trans)

        # Sistema da câmera
        v_camera = (v - eye) @ R.T

        # Homogêneo e projeção
        v_homog = np.hstack([v_camera, np.ones((v_camera.shape[0], 1))])
        v_clip = (P @ v_homog.T).T
        v_ndc = v_clip[:, :3] / v_clip[:, [3]]
        v_2d = v_ndc[:, :2]
        todos_vertices_2d.append(v_2d)

        # Verificar se é objeto 3D ou linha
        if len(obj.faces) > 0 and desenhar_faces:
            # Objeto 3D com faces
            for face in obj.faces:
                if all(idx < len(v_2d) for idx in face):
                    polygon = plt.Polygon(v_2d[face], color=cor, alpha=alpha, linewidth=0.3)
                    ax.add_patch(polygon)
        else:
            # Linha ou pontos
            if len(v_2d) == 2:
                # Linha simples
                ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=2)
                ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=30)
            else:
                # Linha poligonal ou pontos
                ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=1)
                ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=10)

    # Calcular limites dinâmicos (mesmo código anterior)
    if todos_vertices_2d:
        todos_vertices_2d = np.vstack(todos_vertices_2d)
        x_min, y_min = np.min(todos_vertices_2d, axis=0)
        x_max, y_max = np.max(todos_vertices_2d, axis=0)
        
        x_size = x_max - x_min
        y_size = y_max - y_min
        max_size = max(x_size, y_size)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_size/2 * (1 + padding)
        x_max = x_center + max_size/2 * (1 + padding)
        y_min = y_center - max_size/2 * (1 + padding)
        y_max = y_center + max_size/2 * (1 + padding)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ax.set_aspect('equal')
    ax.set_title("Projeção Perspectiva 2D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def rasterizar_objeto(objeto, resolucao='media'):
    # Define parâmetros de resolução para cada tipo de objeto
    if isinstance(objeto, Cano):
        if resolucao == 'baixa':
            return Cano(objeto.vertices[0][0], objeto.vertices[-1][2], 
                       espessura=objeto.vertices[0][0]-objeto.vertices[1][0], 
                       n_segmentos=8, n_cortes=5)
        elif resolucao == 'media':
            return Cano(objeto.vertices[0][0], objeto.vertices[-1][2], 
                       espessura=objeto.vertices[0][0]-objeto.vertices[1][0], 
                       n_segmentos=16, n_cortes=10)
        else:  # alta
            return objeto  # Mantém a resolução original
        
    elif isinstance(objeto, CanoCurvo):
        if resolucao == 'baixa':
            return CanoCurvo(raio_externo=objeto.raio_externo, 
                           comprimento=objeto.comprimento,
                           espessura=objeto.espessura,
                           n_segmentos=8)
        elif resolucao == 'media':
            return CanoCurvo(raio_externo=objeto.raio_externo, 
                           comprimento=objeto.comprimento,
                           espessura=objeto.espessura,
                           n_segmentos=16)
        else:  # alta
            return objeto
            
    elif isinstance(objeto, Cilindro):
        if resolucao == 'baixa':
            return Cilindro(raio=objeto.vertices[0][0], 
                          altura=objeto.vertices[-1][2],
                          n=8, m=5)
        elif resolucao == 'media':
            return Cilindro(raio=objeto.vertices[0][0], 
                          altura=objeto.vertices[-1][2],
                          n=16, m=10)
        else:  # alta
            return objeto
            
    elif isinstance(objeto, LinhaReta):
        # Para linhas, podemos adicionar pontos intermediários baseados na resolução
        comprimento = np.linalg.norm(objeto.vertices[1] - objeto.vertices[0])
        
        if resolucao == 'baixa':
            # Mantém apenas os pontos inicial e final
            return objeto
        elif resolucao == 'media':
            # Adiciona 1 ponto intermediário
            v = np.array([
                objeto.vertices[0],
                (objeto.vertices[0] + objeto.vertices[1]) / 2,
                objeto.vertices[1]
            ])
            return LinhaReta.from_vertices(v)
        else:  # alta
            # Adiciona vários pontos intermediários (5 segmentos)
            v = np.array([
                objeto.vertices[0] + (objeto.vertices[1] - objeto.vertices[0]) * i/5 
                for i in range(6)
            ])
            return LinhaReta.from_vertices(v)
    
        
    elif isinstance(objeto, Paralelepipedo):
        # Extrai as dimensões reais do objeto
        vertices = np.array(objeto.vertices)
        largura = np.max(vertices[:,0]) - np.min(vertices[:,0])
        altura = np.max(vertices[:,2]) - np.min(vertices[:,2])
        espessura = np.max(vertices[:,1]) - np.min(vertices[:,1])
        
        if resolucao == 'baixa':
            # Garante pelo menos 2 divisões em cada eixo para manter a 3D
            return Paralelepipedo(largura=largura,
                                altura=altura,
                                espessura=espessura,
                                n=1, m=1, l=2)
        elif resolucao == 'media':
            # Aumenta um pouco mais a resolução
            return Paralelepipedo(largura=largura,
                                altura=altura,
                                espessura=espessura,
                                n=2, m=2, l=2)
        else:  # alta
            return objeto
            
    else:
        raise ValueError("Tipo de objeto não suportado")

def visualizar_rasterizacoes(objeto_original, nome_objeto):
    fig = plt.figure(figsize=(18, 6))
    
    resolucoes = ['baixa', 'media', 'alta']
    
    for i, resolucao in enumerate(resolucoes, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        objeto = rasterizar_objeto(objeto_original, resolucao)
        vertices = np.array(objeto.vertices)
        
        # Verifica se é linha ou objeto 3D
        if len(objeto.faces) > 0:
            # Objeto 3D
            faces = np.array(objeto.faces)
            mesh = Poly3DCollection(vertices[faces],
                                  alpha=0.8,
                                  linewidths=0.3,
                                  edgecolor='k',
                                  facecolor='skyblue')
            ax.add_collection3d(mesh)
            titulo = f"Vértices: {len(vertices)}, Faces: {len(faces)}"
        else:
            # Linha
            ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], 
                   color='blue', linewidth=2, marker='o')
            titulo = f"Pontos: {len(vertices)}"
        
        # Ajustar visualização
        todos_v = np.vstack(vertices)
        centro = np.mean(todos_v, axis=0)
        max_dist = np.max(np.linalg.norm(todos_v - centro, axis=1)) * 1.5
        
        ax.set_xlim(centro[0]-max_dist, centro[0]+max_dist)
        ax.set_ylim(centro[1]-max_dist, centro[1]+max_dist)
        ax.set_zlim(centro[2]-max_dist, centro[2]+max_dist)
        
        ax.set_title(f"{nome_objeto} - Resolução {resolucao.capitalize()}\n{titulo}")
    
    plt.tight_layout()
    plt.show()