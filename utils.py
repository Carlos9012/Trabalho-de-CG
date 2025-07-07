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