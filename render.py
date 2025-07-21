"""
render.py  –  Transformações, câmera, projeção e visualização rápida.

Depende de:
• numpy
• matplotlib (apenas para debug / visual)
• utils2.py  (fornecido pelo professor – mantém TODA a álgebra)
"""

from __future__ import annotations
import traceback
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button
import matplotlib as mpl
import utils2 as ut
from geometry import Mesh


# ------------------------------------------------------------------ #
#  Utilitário: botão de ligar/desligar arestas (malha)               #
# ------------------------------------------------------------------ #
def _attach_toggle_edges_button(ax, collections,
                                initial_label="Ocultar malha"):
    """
    Botão para alternar a exibição de ARESTAS das superfícies.

    • Poly3DCollection / PolyCollection → muda edgecolor & linewidth
    • Lines são ignoradas (continuam visíveis)
    """
    btn_ax = ax.figure.add_axes([0.83, 0.9, 0.14, 0.065])
    btn    = Button(btn_ax, initial_label,
                    color="lightgray", hovercolor="0.8")

    state = {"show": True}

    def toggle(_event):
        try:
            state["show"] = not state["show"]
            btn.label.set_text("Mostrar malha" if not state["show"]
                               else "Ocultar malha")

            for c in collections:
                if isinstance(c, (Poly3DCollection,
                                  mpl.collections.PolyCollection)):
                    c.set_edgecolor("k" if state["show"] else "none")
                    c.set_linewidth(0.4 if state["show"] else 0.0)
                # linhas (Line3D / Line2D) ficam sempre visíveis
            ax.figure.canvas.draw_idle()

        except Exception as e:
            print("\n[Toggle-Malha] Erro inesperado ao alternar malha:")
            traceback.print_exc()

    btn.on_clicked(toggle)
    return btn

# ------------------------------------------------------------------ #
#  Transformações básicas                                            #
# ------------------------------------------------------------------ #

def model_matrix(scale: float = 1.0,
                 rotation_4x4: np.ndarray | None = None,
                 translation: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """Cria matriz Model = T · R · S."""
    s = ut.matriz_escala(scale, scale, scale)
    t = ut.matriz_translacao(*translation)
    r = rotation_4x4 if rotation_4x4 is not None else np.eye(4)
    return t @ r @ s


def transform_mesh(mesh: Mesh, M: np.ndarray) -> Mesh:
    v = ut.transformar_pontos(mesh.vertices, M)
    return Mesh(v, mesh.faces)


# ------------------------------------------------------------------ #
#  Câmera e projeção                                                 #
# ------------------------------------------------------------------ #

def view_matrix(eye: np.ndarray,
                at:  np.ndarray,
                up:  np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    return ut.matriz_visao(eye, at, up)


def projection_matrix(fov: float, aspect: float,
                      near: float = 1.0, far: float = 50.0) -> np.ndarray:
    return ut.matriz_projecao_perspectiva(fov, aspect, near, far)


def project_vertices(vertices: np.ndarray,
                     M_vp: np.ndarray) -> np.ndarray:
    """Transforma para clip-space → NDC → 2D (x, y)."""
    v_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    v_clip = (M_vp @ v_h.T).T
    v_ndc = v_clip[:, :3] / (v_clip[:, [3]] + 1e-8)
    return v_ndc[:, :2]


# ------------------------------------------------------------------ #
#  Vizualização rápida                                               #
# ------------------------------------------------------------------ #

def plot_scene_3d(
    meshes: list[Mesh],
    colors: list[str] | None = None,
    title: str = "Mundo 3D",
    show_edges: bool = True,
    camera_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    world_origin_cam: np.ndarray | None = None,
) -> None:
    """
    Plota meshes 3-D.

    Parâmetros extras:
    • camera_pose=(eye, at, up) → desenha vetores N U V e posição da câmera.
    • world_origin_cam          → ponto (x,y,z) da origem do mundo APÓS
                                  a transformação pela matriz-view (coordenadas
                                  da câmera); marcado como bolinha preta.
    """
    colors = colors or ["skyblue"] * len(meshes)
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    # ---------- objetos ----------
    for mesh, c in zip(meshes, colors):
        if len(mesh.faces):
            surf = Poly3DCollection(mesh.vertices[mesh.faces],
                                    facecolor=c, alpha=0.65,
                                    edgecolor="k" if show_edges else "none",
                                    linewidths=0.4 if show_edges else 0.0)
            ax.add_collection3d(surf)
        else:
            ax.plot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    color=c, lw=2)

    # ---------- extras ----------
    if camera_pose is not None:
        eye, at, up = camera_pose
        n = (eye - at); n /= np.linalg.norm(n)
        u = np.cross(up, n); u /= np.linalg.norm(u)
        v = np.cross(n, u)

        L = 2.0  # comprimento das setas
        ax.quiver(0, 0, 0,  n[0], n[1], n[2], color="r", length=L, label="n")
        ax.quiver(0, 0, 0,  u[0], u[1], u[2], color="g", length=L, label="u")
        ax.quiver(0, 0, 0,  v[0], v[1], v[2], color="cyan", length=L, label="v")
        ax.scatter(0, 0, 0, color="blue", s=40, label="câmera (0,0,0)")

    if world_origin_cam is not None:
        ax.scatter(*world_origin_cam, color="k", s=40,
                   label="Origem (world)")

    # ---------- limites ----------
    all_v = np.vstack([m.vertices for m in meshes])
    m = np.max(np.abs(all_v)) * 1.2
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    if camera_pose is not None or world_origin_cam is not None:
        ax.legend()

    plt.show()



def plot_projection_2d(
    meshes_2d: list["Mesh"],
    colors: list[str] | None = None,
    title: str = "Projeção 2D",
    filled: bool = False,          # << NOVO parâmetro
    edge_lw: float = 1.0,          # largura da aresta quando filled=True
    alpha: float = 0.5             # transparência dos preenchidos
) -> None:
    """
    Exibe projeção em 2D.

    • filled=False  →  só as arestas (com `edge_lw` ignorado).
    • filled=True   →  triângulos preenchidos + contorno fino.
    """
    colors = colors or ["k"] * len(meshes_2d)
    fig, ax = plt.subplots(figsize=(6, 6))

    for mesh, c in zip(meshes_2d, colors):
        v = mesh.vertices
        if len(mesh.faces) == 0:
            # malha-linha
            ax.plot(v[:, 0], v[:, 1],
                    color=c, lw=edge_lw if filled else 1.5)
            continue

        if filled:
            # ----- preenchido (scan-line já foi feito em raster, aqui é só plot) -----
            polys = [v[idx] for idx in mesh.faces]  # lista (N_i, 2)
            coll = PolyCollection(
                polys,
                facecolors=c,
                edgecolors="k",
                linewidths=edge_lw,
                alpha=alpha
            )
            ax.add_collection(coll)
        else:
            # ----- apenas arestas -----
            for tri in mesh.faces:
                loop = np.append(tri, tri[0])
                ax.plot(v[loop, 0], v[loop, 1], color=c, lw=1)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)
    plt.show()
    
    
    
# ------------------------------------------------------------------ #
#  Plotar objetos isolados                                           #
# ------------------------------------------------------------------ #
def plot_objects_isolated(meshes: list[Mesh],
                          colors: list[str] | None = None,
                          title: str = "Objetos – sistema local") -> None:
    """
    Plota cada mesh na origem (sem transformações).
    Todos os objetos no mesmo subplot (afastados em X para visual).
    """
    colors = colors or ["skyblue"] * len(meshes)
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(left=0.07, right=0.80, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111, projection="3d")
    coll = []

    offset = 0.0
    gap    = 4.0  # distância entre objetos
    for mesh, c in zip(meshes, colors):
        verts = mesh.vertices.copy()
        verts[:, 0] += offset       # desloca no X
        offset += gap

        if len(mesh.faces):
            pc = Poly3DCollection(verts[mesh.faces],
                                  facecolor=c, alpha=0.65,
                                  edgecolor="k", linewidths=0.4)
            ax.add_collection3d(pc)
            coll.append(pc)
        else:
            line, = ax.plot(verts[:,0], verts[:,1], verts[:,2],
                            color=c, lw=2)
            coll.append(line)

    ax.set_box_aspect([offset, gap, gap])
    ax.set_title(title)
    _attach_toggle_edges_button(ax, coll,
                                initial_label="Ocultar malha")
    plt.show()

