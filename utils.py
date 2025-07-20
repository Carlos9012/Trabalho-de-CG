import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from objects import Cano, CanoCurvo, Cilindro, LinhaReta, Paralelepipedo
import utils2 as fgl

def _transformar_vertices_objeto(vertices, escala, matriz_rotacao_4x4, translacao):
    """Aplica uma sequência completa de transformações a um conjunto de vértices."""
    # Cria as matrizes de transformação individuais
    M_esc = fgl.matriz_escala(escala, escala, escala)
    M_trans = fgl.matriz_translacao(translacao[0], translacao[1], translacao[2])

    # Combina as transformações na ordem correta
    M_modelo = M_trans @ matriz_rotacao_4x4 @ M_esc

    # Aplica a matriz final
    return fgl.transformar_pontos(vertices, M_modelo)

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

def compor_cena_varios(objetos_transformados: list[tuple], mostrar_malha: bool = True, title: str = "Cena 3D com múltiplos objetos", tam_max: float = 10.0):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    todos_vertices, legend_elements = [], []

    for i, (obj, esc, rot, trans, cor) in enumerate(objetos_transformados):
        # AJUSTE: Converte a matriz de rotação 3x3 para 4x4 se necessário
        if rot.shape == (3, 3):
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot
        elif rot.shape == (4, 4):
            rot_4x4 = rot
        else:
            raise ValueError(f"Matriz de rotação deve ser 3x3 ou 4x4, recebida {rot.shape}")

        v = _transformar_vertices_objeto(np.array(obj.vertices), esc, rot_4x4, trans)

        max_extent = np.max(np.linalg.norm(v, axis=1)) * 1.2
        if max_extent > tam_max:
            esc *= tam_max / max_extent
            v = _transformar_vertices_objeto(np.array(obj.vertices), esc, rot_4x4, trans)

        todos_vertices.append(v)
        if len(obj.faces) > 0:
            mesh = Poly3DCollection(v[np.array(obj.faces)], alpha=0.6, linewidths=0.3 if mostrar_malha else 0, edgecolor='k' if mostrar_malha else 'none', facecolor=cor)
            ax.add_collection3d(mesh)
        else:
            if len(v) == 2: ax.plot(v[:,0], v[:,1], v[:,2], color=cor, linewidth=2, marker='o', markersize=6)
            else: ax.plot(v[:,0], v[:,1], v[:,2], color=cor, linewidth=2)
        legend_elements.append(Patch(facecolor=cor, label=f'Obj {i+1}'))

    if todos_vertices:
        todos_vertices = np.vstack(todos_vertices)
        centro = np.mean(todos_vertices, axis=0)
        max_dist = np.max(np.linalg.norm(todos_vertices - centro, axis=1)) * 1.5
    else:
        centro, max_dist = np.array([0,0,0]), 10.0
    ax.set_xlim(centro[0]-max_dist, centro[0]+max_dist); ax.set_ylim(centro[1]-max_dist, centro[1]+max_dist); ax.set_zlim(centro[2]-max_dist, centro[2]+max_dist)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(f"{title}\nEscala máxima: {tam_max}"); ax.legend(handles=legend_elements)
    plt.tight_layout(); plt.show()

def mostrar_cena_camera_3D(objetos, eye, at, up=np.array([0, 1, 0]), mostrar_malha=True):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    M_visao = fgl.matriz_visao(eye, at, up)

    todos_vertices_camera = []
    legend_elements = []

    for i, (obj, esc, rot, trans, cor) in enumerate(objetos):
        # AJUSTE: Garante que a matriz de rotação seja 4x4
        if rot.shape == (3, 3):
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot
        elif rot.shape == (4, 4):
            rot_4x4 = rot
        else:
            raise ValueError(f"Matriz de rotação deve ser 3x3 ou 4x4, recebida {rot.shape}")

        # Matriz de modelo completa
        M_modelo = fgl.matriz_translacao(*trans) @ rot_4x4 @ fgl.matriz_escala(esc, esc, esc)

        # Transformação para coordenadas da câmera
        v_mundo = fgl.transformar_pontos(np.array(obj.vertices), M_modelo)
        v_camera = fgl.transformar_pontos(v_mundo, M_visao)
        todos_vertices_camera.append(v_camera)

        # Renderização
        if len(obj.faces) > 0:
            mesh = Poly3DCollection(
                v_camera[np.array(obj.faces)],
                alpha=0.6,
                linewidths=0.3 if mostrar_malha else 0,
                edgecolor='k' if mostrar_malha else 'none',
                facecolor=cor
            )
            ax.add_collection3d(mesh)
        else:
            if len(v_camera) == 2:
                ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2],
                       color=cor, linewidth=2, marker='o', markersize=6)
            else:
                ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2],
                       color=cor, linewidth=2)

        legend_elements.append(Patch(facecolor=cor, label=f'Obj {i+1}'))

    # Restante da função permanece o mesmo...
    origem_mundo = np.array([0, 0, 0])
    origem_mundo_camera = fgl.transformar_pontos(np.array([origem_mundo]), M_visao)[0]
    ax.scatter(origem_mundo_camera[0], origem_mundo_camera[1], origem_mundo_camera[2], color='red', s=100, label='Origem do Mundo')
    ax.scatter(0, 0, 0, color='blue', s=100, label='Origem da Câmera')
    ax.plot([0, origem_mundo_camera[0]], [0, origem_mundo_camera[1]], [0, origem_mundo_camera[2]], 'k--', alpha=0.5, label='Linha Mundo-Câmera')

    if todos_vertices_camera:
        all_v = np.vstack(todos_vertices_camera)
        centro = np.mean(all_v, axis=0)
        max_dist = np.max(np.linalg.norm(all_v - centro, axis=1)) * 1.5
    else:
        centro, max_dist = np.array([0,0,0]), 10.0

    ax.set_xlim(centro[0]-max_dist, centro[0]+max_dist)
    ax.set_ylim(centro[1]-max_dist, centro[1]+max_dist)
    ax.set_zlim(centro[2]-max_dist, centro[2]+max_dist)
    ax.set_title("Objetos no Sistema da Câmera (U, V, N)")
    ax.set_xlabel("U (Direita)")
    ax.set_ylabel("V (Cima)")
    ax.set_zlabel("N (Profundidade)")
    plt.tight_layout()
    plt.show()

def projetar_perspectiva_2d(objetos, eye, at, up=np.array([0, 1, 0]), fov=60, aspect_ratio=1, near=1, far=100, desenhar_faces=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    M_visao = fgl.matriz_visao(eye, at, up)
    M_proj = fgl.matriz_projecao_perspectiva(fov, aspect_ratio, near, far)

    todos_vertices_2d = []
    for obj, esc, rot, trans, cor in objetos:
        # AJUSTE: Garante que a matriz de rotação seja 4x4
        if rot.shape == (3, 3):
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot
        elif rot.shape == (4, 4):
            rot_4x4 = rot
        else:
            raise ValueError(f"Matriz de rotação deve ser 3x3 ou 4x4, recebida {rot.shape}")

        M_modelo = fgl.matriz_translacao(*trans) @ rot_4x4 @ fgl.matriz_escala(esc, esc, esc)
        M_final = M_proj @ M_visao @ M_modelo

        v_homog = np.hstack([np.array(obj.vertices), np.ones((len(obj.vertices), 1))])
        v_clip = (M_final @ v_homog.T).T
        v_ndc = v_clip[:, :3] / (v_clip[:, [3]] + 1e-8)
        v_2d = v_ndc[:, :2]
        todos_vertices_2d.append(v_2d)

        # Renderização
        if len(obj.faces) > 0 and desenhar_faces:
            for face in obj.faces:
                polygon = plt.Polygon(v_2d[np.array(face)], color=cor, alpha=0.6, linewidth=0.3)
                ax.add_patch(polygon)
        else:
            if len(v_2d) == 2:
                ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=2)
                ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=30)
            else:
                ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=1)
                ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=10)

    # Restante da função permanece o mesmo...
    if todos_vertices_2d:
        all_v = np.vstack(todos_vertices_2d)
        x_min, y_min = np.min(all_v, axis=0)
        x_max, y_max = np.max(all_v, axis=0)
        ax.set_xlim(x_min-0.1, x_max+0.1)
        ax.set_ylim(y_min-0.1, y_max+0.1)
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

def rasterizar_cena(objetos, eye, at, up, resolucao):
    largura, altura = resolucao
    aspect_ratio = largura / altura

    M_visao = fgl.matriz_visao(eye, at, up)
    M_proj = fgl.matriz_projecao_perspectiva(60, aspect_ratio, 1, 100)

    cena_rasterizada = []

    for obj, esc, rot, trans, cor in objetos:
        # AJUSTE: Garante que a matriz de rotação seja 4x4
        if rot.shape == (3, 3):
            rot_4x4 = np.eye(4)
            rot_4x4[:3, :3] = rot
        elif rot.shape == (4, 4):
            rot_4x4 = rot
        else:
            raise ValueError("Matriz de rotação deve ser 3x3 ou 4x4")

        M_modelo = fgl.matriz_translacao(*trans) @ rot_4x4 @ fgl.matriz_escala(esc, esc, esc)
        
        v_homog = np.hstack([obj.vertices, np.ones((len(obj.vertices), 1))])
        v_ndc = (M_proj @ M_visao @ M_modelo @ v_homog.T).T
        v_ndc = v_ndc[:, :3] / (v_ndc[:, [3]] + 1e-8)

        v_2d = (v_ndc[:, :2] + 1) * 0.5 * np.array([largura, altura])
        v_2d = v_2d.astype(int)

        pixels = []
        if len(obj.faces) > 0:
            for face in obj.faces:
                vertices_face = [v_2d[i] for i in face]
                pixels.extend(fgl.rasterizar_poligono_scanline(vertices_face))
        elif len(v_2d) >= 2:
            for i in range(len(v_2d) - 1):
                pixels.extend(fgl.rasterizar_linha_bresenham(v_2d[i], v_2d[i+1]))
        
        pixels = [(x, y) for x, y in pixels if 0 <= x < largura and 0 <= y < altura]
        cena_rasterizada.append((pixels, cor))

    return cena_rasterizada

def visualizar_rasterizacao(cena_raster, resolucao, title="Cena Rasterizada"):
    imagem = np.zeros((resolucao[1], resolucao[0], 3))
    
    cores = {
        'lightblue': [173/255, 216/255, 230/255],
        'salmon': [250/255, 128/255, 114/255],
        'khaki': [240/255, 230/255, 140/255],
        'red': [1.0, 0.0, 0.0],
        'lightgreen': [144/255, 238/255, 144/255]
    }
    
    for pixels, cor in cena_raster:
        cor_rgb = np.array(cores.get(cor, [1, 1, 1]))
        for x, y in pixels:
            if 0 <= x < resolucao[0] and 0 <= y < resolucao[1]:
                imagem[y, x] = cor_rgb

    plt.figure(figsize=(10, 8))
    plt.imshow(imagem)
    plt.title(title)
    plt.axis('off')
    plt.show()