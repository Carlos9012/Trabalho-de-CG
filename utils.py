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

    for i, (obj, esc, rot_3x3, trans, cor) in enumerate(objetos_transformados):
        # Converte a matriz de rotação 3x3 para 4x4 para ser compatível com as outras
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot_3x3
        
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

    # 1. Usa a matriz de visão NUV de utils2.py
    M_visao = fgl.matriz_visao(eye, at, up)
    
    # Extrai os eixos da câmera da parte de rotação da matriz
    n_axis, u_axis, v_axis = M_visao[0, :3], M_visao[1, :3], M_visao[2, :3]

    todos_vertices_camera = []
    legend_elements = [] # Inicia a lista de legendas

    for i, (obj, esc, rot_3x3, trans, cor) in enumerate(objetos):
        rot_4x4 = np.eye(4); rot_4x4[:3, :3] = rot_3x3
        M_modelo = fgl.matriz_translacao(*trans) @ rot_4x4 @ fgl.matriz_escala(esc, esc, esc)
        
        v_mundo_transformado = fgl.transformar_pontos(np.array(obj.vertices), M_modelo)
        v_camera = fgl.transformar_pontos(v_mundo_transformado, M_visao)
        todos_vertices_camera.append(v_camera)
        
        if len(obj.faces) > 0:
            mesh = Poly3DCollection(v_camera[np.array(obj.faces)], alpha=0.6, linewidths=0.3 if mostrar_malha else 0, edgecolor='k' if mostrar_malha else 'none', facecolor=cor)
            ax.add_collection3d(mesh)
        else:
            if len(v_camera) == 2: ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2], color=cor, linewidth=2, marker='o', markersize=6)
            else: ax.plot(v_camera[:,0], v_camera[:,1], v_camera[:,2], color=cor, linewidth=2)
        
        # Adiciona a legenda para cada objeto
        legend_elements.append(Patch(facecolor=cor, label=f'Obj {i+1}'))

    # --- CÓDIGO ADICIONADO PARA PLOTAR AS ORIGENS ---
    
    # A origem da câmera no sistema da câmera é sempre (0,0,0)
    origem_camera = np.array([0, 0, 0])
    
    # A origem do mundo é (0,0,0) no sistema do mundo.
    # Para saber onde ela está no sistema da câmera, aplicamos a matriz de visão.
    origem_mundo_na_camera = fgl.transformar_pontos(np.array([[0, 0, 0]]), M_visao)[0]
    
    # Plota a origem da câmera (ponto azul)
    ax.scatter(*origem_camera, color='blue', s=100, label='Origem Câmera (eye)')
    
    # Plota a origem do mundo (ponto vermelho)
    ax.scatter(*origem_mundo_na_camera, color='red', s=100, label='Origem Mundo (0,0,0)')
    
    # Plota a linha tracejada conectando as duas origens
    ax.plot(*zip(origem_camera, origem_mundo_na_camera), 'k--', alpha=0.5, label='Mundo -> Câmera')
    
    # --- FIM DO CÓDIGO ADICIONADO ---

    # Lógica de ajuste dos limites (inalterada)
    if todos_vertices_camera:
        all_v = np.vstack(todos_vertices_camera)
        centro = np.mean(all_v, axis=0)
        max_dist = np.max(np.linalg.norm(all_v - centro, axis=1)) * 1.5 if all_v.size > 0 else 10.0
    else:
        centro, max_dist = np.array([0,0,0]), 10.0
        
    ax.set_xlim(centro[0]-max_dist, centro[0]+max_dist); ax.set_ylim(centro[1]-max_dist, centro[1]+max_dist); ax.set_zlim(centro[2]-max_dist, centro[2]+max_dist)
    ax.set_title("Objetos no Sistema da Câmera (NUV)")
    
    # Desenho dos eixos e legendas
    ax.set_xlabel("N (Profundidade)"); ax.set_ylabel("U (Altura)"); ax.set_zlabel("V (Lateral)")
    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], n_axis[0], n_axis[1], n_axis[2], color='blue', length=max_dist/5, label='Eixo N (View Dir)')
    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], u_axis[0], u_axis[1], u_axis[2], color='green', length=max_dist/5, label='Eixo U (Up)')
    ax.quiver(origem_camera[0], origem_camera[1], origem_camera[2], v_axis[0], v_axis[1], v_axis[2], color='red', length=max_dist/5, label='Eixo V (Right)')
    
    # Combina todas as legendas (objetos + eixos + origens)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + [h.get_label() for h in legend_elements])
    
    plt.tight_layout(); plt.show()

def projetar_perspectiva_2d(objetos, eye, at, up=np.array([0, 1, 0]), fov=60, aspect_ratio=1, near=1, far=100, desenhar_faces=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    # SUBSTITUIÇÃO: Calcula as matrizes de Visão e Projeção chamando utils2.py
    M_visao = fgl.matriz_visao(eye, at, up)
    M_proj = fgl.matriz_projecao_perspectiva(fov, aspect_ratio, near, far)
    
    todos_vertices_2d = []
    for obj, esc, rot_3x3, trans, cor in objetos:
        rot_4x4 = np.eye(4); rot_4x4[:3, :3] = rot_3x3
        
        # 1. Matriz do Modelo
        M_modelo = fgl.matriz_translacao(*trans) @ rot_4x4 @ fgl.matriz_escala(esc, esc, esc)
        
        # 2. Pipeline completo: Modelo -> Visão -> Projeção
        M_final = M_proj @ M_visao @ M_modelo
        
        v_homog = np.hstack([np.array(obj.vertices), np.ones((len(obj.vertices), 1))])
        v_clip = (M_final @ v_homog.T).T
        
        # 3. Divisão de Perspectiva para obter NDC
        v_ndc = v_clip[:, :3] / (v_clip[:, [3]] + 1e-8)
        
        v_2d = v_ndc[:, :2]
        todos_vertices_2d.append(v_2d)
        
        if len(obj.faces) > 0 and desenhar_faces:
            for face in obj.faces:
                polygon = plt.Polygon(v_2d[np.array(face)], color=cor, alpha=0.6, linewidth=0.3)
                ax.add_patch(polygon)
        else:
            if len(v_2d) == 2: ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=2); ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=30)
            else: ax.plot(v_2d[:, 0], v_2d[:, 1], color=cor, linewidth=1); ax.scatter(v_2d[:, 0], v_2d[:, 1], color=cor, s=10)

    if todos_vertices_2d:
        all_v = np.vstack(todos_vertices_2d)
        x_min, y_min = np.min(all_v, axis=0); x_max, y_max = np.max(all_v, axis=0)
        ax.set_xlim(x_min-0.1, x_max+0.1); ax.set_ylim(y_min-0.1, y_max+0.1)
    else:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    
    ax.set_aspect('equal'); ax.set_title("Projeção Perspectiva 2D"); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.grid(True); plt.tight_layout(); plt.show()

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