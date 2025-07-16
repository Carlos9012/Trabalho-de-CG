import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch 


def _transformar_vertices(vertices, escala=1.0, rotacao=np.eye(3), translacao=np.zeros(3)):
    S = np.diag([escala, escala, escala])
    return (vertices @ S @ rotacao.T) + translacao

def plotar(objeto, mostrar_malha=True, cor='skyblue', title='Figura'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    v = np.array(objeto.vertices)
    f = np.array(objeto.faces)

    mesh = Poly3DCollection(
        v[f],
        alpha=0.6,
        linewidths=0.3 if mostrar_malha else 0,
        edgecolor='k' if mostrar_malha else 'none',
        facecolor=cor
    )
    ax.add_collection3d(mesh)

    todos_v = np.vstack(v)
    max_c = np.max(np.abs(todos_v)) * 1.1
    ax.set_xlim(-max_c, max_c)
    ax.set_ylim(-max_c, max_c)
    ax.set_zlim(0, max_c)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
       
def compor_cena_varios(objetos_transformados: list[tuple],
                      mostrar_malha: bool = True,
                      title: str = "Cena 3D com múltiplos objetos",
                      save_path: [str] = None,
                      tam_max: float = 10.0,
                      debug: bool = False) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    tamanhos_maximos = []
    offset = 0
    todos_vertices = []
    legend_elements = []

    for i, (obj, esc, rot, trans, cor) in enumerate(objetos_transformados):
        vertices_originais = np.array(obj.vertices)

        # Transformar inicialmente
        v = _transformar_vertices(vertices_originais, esc, rot, trans)
        max_extent = np.max(np.linalg.norm(v, axis=1)) *1.2

        # Reduzir escala se exceder tam_max
        if max_extent  > tam_max:
            fator_escala = tam_max / max_extent
            esc = esc * fator_escala
            v = _transformar_vertices(vertices_originais, esc, rot, trans)
            max_extent = np.max(np.linalg.norm(v, axis=1.4))
            if debug:
                print(f"Objeto {i+1} reescalado para {fator_escala:.2f}x para caber em tam_max")

        tamanhos_maximos.append((i, max_extent))

        # Verificar e ajustar índices das faces
        faces_validas = []
        for face in obj.faces:
            if all(idx < len(v) for idx in face):
                faces_validas.append(face)
            elif debug:
                print(f"Aviso: Face inválida {face} removida (vértices máx: {len(v)-1})")

        f = np.array(faces_validas) + offset

        mesh = Poly3DCollection(
            v[f - offset],
            alpha=0.6,
            linewidths=0.3 if mostrar_malha else 0,
            edgecolor='k' if mostrar_malha else 'none',
            facecolor=cor
        )
        ax.add_collection3d(mesh)

        todos_vertices.append(v)
        offset += len(v)
        legend_elements.append(Patch(facecolor=cor, label=f'Obj {i+1} (max={max_extent:.1f})'))

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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

    offset = 0
    todos_vertices = []

    # Plotar objetos
    for obj, esc, rot, trans, cor in objetos:
        v_world = _transformar_vertices(np.array(obj.vertices), esc, rot, trans)
        v_camera = np.dot(v_world - eye, R.T)

        f = np.array(obj.faces) + offset
        mesh = Poly3DCollection(
            v_camera[f - offset],
            alpha=0.6,
            linewidths=0.3 if mostrar_malha else 0,
            edgecolor='k' if mostrar_malha else 'none',
            facecolor=cor
        )
        ax.add_collection3d(mesh)
        todos_vertices.append(v_camera)
        offset += len(v_camera)

    todos_vertices = np.vstack(todos_vertices)
    centro = np.mean(todos_vertices, axis=0)
    max_dist = np.max(np.linalg.norm(todos_vertices - centro, axis=1)) * 1.5

    # Origem do sistema do mundo no sistema da câmera
    origem_mundo_camera = np.dot(-eye, R.T)

    # Origem da câmera no sistema da câmera (é o ponto [0,0,0])
    origem_camera = eye

    # Linha conectando os sistemas de coordenadas
    if mostrar_linha_camera:
        ax.plot([origem_camera[0], origem_mundo_camera[0]],
                [origem_camera[1], origem_mundo_camera[1]],
                [origem_camera[2], origem_mundo_camera[2]],
                'k--', linewidth=1, alpha=0.5, label='Sistema Mundo→Câmera')

    # Visualização dos eixos e pontos de origem
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

def projetar_perspectiva_2d(objetos,
                          eye,
                          at,
                          up=np.array([0, 1, 0]),
                          fov=60,
                          aspect_ratio=1,
                          near=1,
                          far=100,
                          desenhar_faces=True,
                          alpha=0.6,
                          padding=0.1):
    """
    Projeta objetos 3D em uma vista 2D utilizando projeção perspectiva com faces coloridas.
    
    Parâmetros:
    - objetos: Lista de tuplas (objeto, escala, rotação, translação, cor)
    - eye: Posição da câmera
    - at: Ponto para onde a câmera está olhando
    - up: Vetor up da câmera
    - fov: Campo de visão em graus
    - aspect_ratio: Proporção da tela
    - near: Plano near
    - far: Plano far
    - desenhar_faces: Se True, desenha faces preenchidas
    - alpha: Transparência das faces (0-1)
    - padding: Espaço adicional ao redor dos objetos (fração do tamanho total)
    """
    # Passo 1: base da câmera (view matrix)
    n = (eye - at)
    n = n / (np.linalg.norm(n) + 1e-8)
    
    if np.allclose(np.cross(up, n), 0):
        up = np.array([0, 0, 1]) if not np.allclose(up, [0, 0, 1]) else np.array([1, 0, 0])
    
    u = np.cross(up, n)
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    R = np.stack([u, v, n], axis=0)

    # Passo 2: construção da matriz de projeção
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
    
    # Lista para armazenar todas as coordenadas 2D
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
        
        # Adiciona vértices à lista geral
        todos_vertices_2d.append(v_2d)

        if desenhar_faces:
            # Desenhar faces preenchidas
            for face in obj.faces:
                if all(idx < len(v_2d) for idx in face):
                    polygon = plt.Polygon(v_2d[face], color=cor, alpha=alpha, linewidth=0.3)
                    ax.add_patch(polygon)
        else:
            ax.scatter(v_2d[:, 0], v_2d[:, 1], s=1, color=cor)

    # Calcular limites dinâmicos
    if todos_vertices_2d:
        todos_vertices_2d = np.vstack(todos_vertices_2d)
        x_min, y_min = np.min(todos_vertices_2d, axis=0)
        x_max, y_max = np.max(todos_vertices_2d, axis=0)
        
        # Calcular tamanho máximo em qualquer direção
        x_size = x_max - x_min
        y_size = y_max - y_min
        max_size = max(x_size, y_size)
        
        # Centralizar e adicionar padding
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - max_size/2 * (1 + padding)
        x_max = x_center + max_size/2 * (1 + padding)
        y_min = y_center - max_size/2 * (1 + padding)
        y_max = y_center + max_size/2 * (1 + padding)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Fallback caso não haja vértices
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ax.set_aspect('equal')
    ax.set_title("Projeção Perspectiva 2D com Faces Coloridas")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()