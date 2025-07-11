import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _transformar_vertices(vertices, escala=1.0, rotacao=np.eye(3), translacao=np.zeros(3)):
    return np.dot(vertices, rotacao.T) * escala + translacao

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
    
    
    
    
def compor_cena_varios(objetos_transformados, mostrar_malha=True):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    offset = 0
    todos_vertices = []

    for obj, esc, rot, trans, cor in objetos_transformados:
        v = _transformar_vertices(np.array(obj.vertices), esc, rot, trans)
        f = np.array(obj.faces) + offset

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

    todos_vertices = np.vstack(todos_vertices)
    max_c = np.max(np.abs(todos_vertices)) * 1.1
    ax.set_xlim(-max_c, max_c)
    ax.set_ylim(-max_c, max_c)
    ax.set_zlim(-max_c, max_c)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cena 3D com múltiplos objetos")
    plt.tight_layout()
    plt.show()

def mostrar_cena_camera_(objetos, camera_pos, look_at, up=np.array([0, 1, 0]), mostrar_malha=True):
    """
    Exibe os objetos no sistema de coordenadas da câmera.

    :param objetos: Lista de tuplas (objeto, escala, rotacao, translacao, cor)
    :param camera_pos: Posição da câmera (np.array)
    :param look_at: Ponto que a câmera está olhando (np.array)
    :param up: Vetor para cima no mundo
    """
    # Base da câmera
    w = (camera_pos - look_at)
    w = w / np.linalg.norm(w)
    u = np.cross(up, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    # Matriz de rotação da câmera (colunas = eixos da base)
    R = np.stack([u, v, w], axis=0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    offset = 0
    todos_vertices = []

    for obj, esc, rot, trans, cor in objetos:
        v_world = _transformar_vertices(np.array(obj.vertices), esc, rot, trans)

        # Transformação para sistema da câmera
        v_camera = np.dot(v_world - camera_pos, R.T)

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
    max_coord = np.max(np.abs(todos_vertices)) * 1.1
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_zlim(-max_coord, max_coord)

    # Origem do sistema do mundo (transformada para sistema da câmera)
    origem_mundo_camera = np.dot(-camera_pos, R.T)
    ax.scatter(*origem_mundo_camera, color='red', s=100, label='Origem do Mundo (0,0,0)')
    ax.legend()

    ax.set_xlabel("u (Right)")
    ax.set_ylabel("v (Up)")
    ax.set_zlabel("w (Backward)")
    ax.set_title("Objetos no Sistema da Câmera")
    plt.tight_layout()
    plt.show()
