"""
example.py – Demonstração completa dos itens 1-5 do Trabalho.

• Modela todos os sólidos pedidos;
• Monta cena (coordenadas do mundo, sem intersecção);
• Converte p/ sistema da câmera, mostra origem do mundo;
• Projeta em perspectiva (arestas coloridas);
• Rasteriza em 3 resoluções diferentes.
"""

import numpy as np, matplotlib.pyplot as plt
import render as rd, raster
import window as win
from geometry import (box, cylinder, line,
                      pipe_straight, pipe_curved, Mesh)

# ------------------------------------------------------------------ #
# 1) modelagem dos sólidos                                           #
# ------------------------------------------------------------------ #
solids   = [
    pipe_straight(radius=0.6, length=4.0, thickness=0.15),
    pipe_curved (radius=0.4, p1=(0,0,0), p2=(3,2,2),
                 t1=(3,0,0), t2=(0,3,0)),
    cylinder    (radius=0.8, height=2.5),
    box         (1.8, 1.2, 1.2),
    line        ((0,0,0), (0,0,4))
]
colors = ["cornflowerblue", "orange", "limegreen",
          "magenta", "red"]            # cor única por sólido

#rd.plot_objects_isolated(solids, colors, title="Objetos isolados (sistema local)")

# ------------------------------------------------------------------ #
# 2) composição da cena (mundo)                                      #
# ------------------------------------------------------------------ #
T = [(-4,0,0),(4,0,0),(0,0,0),(0,5,3.5),(0,-4,0)]
world_meshes: list[Mesh] = []
for m,trans in zip(solids, T):
    M = rd.model_matrix(scale=1.0, translation=trans)
    world_meshes.append(rd.transform_mesh(m, M))

# exibe cena em 3D (mundo) + origem
#rd.plot_scene_3d(world_meshes, colors, "Mundo 3D")
# ------------------------------------------------------------------ #
# 3) câmera – transforma p/ coords da câmera e mostra                #
# ------------------------------------------------------------------ #
eye = np.array([8, 6, 6])
at  = np.array([0, 0, 0])
up  = np.array([0, 1, 0])

V = rd.view_matrix(eye, at, up)
origin_world_cam = (V @ np.array([0, 0, 0, 1])).flatten()[:3]
cam_meshes = [rd.transform_mesh(m, V) for m in world_meshes]
#rd.plot_scene_3d(
#    cam_meshes, colors,
#    title="Sistema da Câmera (3D)",
#    camera_pose=(np.zeros(3), at-eye, up),   # eye==0 no espaço da câmera
#    world_origin_cam=origin_world_cam
#)

# coloca ponto para origem do mundo (vermelho) no último plot
# (apenas estilístico – não necessário em relatório)

# ------------------------------------------------------------------ #
# 4) Projeção em perspectiva (2D)                                    #
# ------------------------------------------------------------------ #
P = rd.projection_matrix(fov=80, aspect=1.0, near=1, far=50)
M_vp = P @ np.eye(4)            # projec. após V já foi aplicada
proj_meshes = [Mesh(rd.project_vertices(m.vertices, P),
                    m.faces) for m in cam_meshes]

#rd.plot_projection_2d(proj_meshes, colors, "Projeção 2D – Arestas")


#rd.plot_projection_2d(proj_meshes, colors, "Projeção 2D - Filled", filled=True)

# ------------------------------------------------------------------ #
# 5) Rasterização em 3 resoluções                                    #
# ------------------------------------------------------------------ #
sizes = [120, 240, 960]
#fig, axes = plt.subplots(1, len(sizes), figsize=(12,4))
#for ax, sz in zip(axes, sizes):
#    fb = raster.rasterize_scene(world_meshes, P@V, sz, sz)
#    ax.imshow(fb, cmap="gray"); ax.set_title(f"{sz}x{sz}")
#    ax.axis("off")
#plt.suptitle("Rasterização (Scan-line) em 3 resoluções")
#plt.tight_layout(); plt.show()


objetos = [
    pipe_straight(radius=0.6, length=4.0, thickness=0.15),
    pipe_curved (radius=0.4, p1=(0,0,0), p2=(3,2,2), t1=(3,0,0), t2=(0,3,0)),
    cylinder    (radius=0.8, height=2.5),
    box         (1.8, 1.2, 1.2),
    line        ((0,0,0), (0,0,4))
]

eye = [5, 5, 5]
at = [0, 0, 0]
up = [0, 1, 0]
fov = 60
aspect_ratio = 1.0
near = 1.0
far = 12.0

win.desenhar_cena(objetos, eye, at, up, fov, aspect_ratio, near, far)