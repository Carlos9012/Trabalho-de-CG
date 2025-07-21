"""
raster.py  –  Pipeline de rasterização em software puro.

• Usa utils2.rasterizar_linha_bresenham  (wireframe)
• Usa utils2.rasterizar_poligono_scanline (preenchido)

Interfaces principais:
    ► ndc_to_screen(v_ndc, W, H)
    ► rasterize_wireframe(meshes, M_vp, W, H)
    ► rasterize_filled(meshes,  M_vp, W, H)

Obs.: Meshes devem ter vértices em espaço do mundo;  M_vp = P @ V.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
import utils2 as ut
import render as rd            # para project_vertices
from geometry import Mesh




# ------------------------------------------------------------------ #
#  Função auxiliar: scan-line “à prova” de KeyError                   #
# ------------------------------------------------------------------ #
def _safe_scanline(verts: List[tuple[float, float]],
                   max_attempts: int = 5,
                   jitter: float = 1e-3):
    """
    Versão robusta: se o polígono projetado tiver altura (ou largura)
    < 1 px, gera manualmente um pequeno retângulo 1×1 px que cobre
    a área, evitando furos em triângulos “rasos”.
    """
    # 1) se o bounding-box tem menos de 1 px em x ou y → retângulo 1×1
    xs, ys = zip(*verts)
    if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
        x0 = int(round(sum(xs)/len(xs)))
        y0 = int(round(sum(ys)/len(ys)))
        return [(x0, y0)]                       # pinta 1 px central

    # 2) tenta normalmente + jitter progressivo
    for _ in range(max_attempts):
        try:
            return ut.rasterizar_poligono_scanline(verts)
        except KeyError:
            verts = [(x, y + jitter) for x, y in verts]
            jitter *= 2
    return []


# ------------------------------------------------------------------ #
#  Utilitário de CLAMP                                               #
# ------------------------------------------------------------------ #
def _clamp_screen(v2d: np.ndarray,
                  width: int,
                  height: int,
                  eps: float = 1e-3) -> np.ndarray:
    """
    Garante 0 ≤ x ≤ W-1-ε e 0 ≤ y ≤ H-1-ε, evitando ceil() fora da tela.
    """
    v2d_clamped = v2d.copy()
    v2d_clamped[:, 0] = np.clip(v2d_clamped[:, 0], 0, width  - 1 - eps)
    v2d_clamped[:, 1] = np.clip(v2d_clamped[:, 1], 0, height - 1 - eps)
    return v2d_clamped


# ------------------------------------------------------------------ #
#  Conversão NDC  →  coordenadas de tela                             #
# ------------------------------------------------------------------ #

def ndc_to_screen(v_ndc: np.ndarray,
                  width: int,
                  height: int) -> np.ndarray:
    """
    Converte vértices NDC (x,y ∈ [-1,+1]) para píxeis da tela:
    (0,0) canto superior-esquerdo, y cresce para baixo.
    """
    x = (v_ndc[:, 0] + 1) * 0.5 * (width  - 1)
    y = (1 - (v_ndc[:, 1] + 1) * 0.5) * (height - 1)
    return np.stack([x, y], axis=1)          # shape (N,2)


# ------------------------------------------------------------------ #
#  Rasterização                                                      #
# ------------------------------------------------------------------ #

def _draw_line(img: np.ndarray,
               p1: Tuple[int, int],
               p2: Tuple[int, int],
               value: int = 255) -> None:
    for x, y in ut.rasterizar_linha_bresenham(p1, p2):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = value



# ------------------------------------------------------------------ #
#  Raster para UM objeto (mesh)                                      #
# ------------------------------------------------------------------ #
def _rasterize_single(mesh: Mesh,
                      M_vp: np.ndarray,
                      width: int,
                      height: int,
                      img: np.ndarray):
    """
    Desenha um único mesh no framebuffer 'img' combinando preenchimento
    e desenho de arestas para evitar buracos.
    """
    v_ndc = rd.project_vertices(mesh.vertices, M_vp)
    v2d = ndc_to_screen(v_ndc, width, height)
    v2d_clamped = _clamp_screen(v2d, width, height)
    v2d_int = v2d.astype(int)

    # Caso 1: A malha é uma linha simples (sem faces)
    if len(mesh.faces) == 0:
        if len(v2d_int) >= 2:
            p1, p2 = map(tuple, v2d_int)
            _draw_line(img, p1, p2)
        return

    # Caso 2: A malha é um sólido (com faces)
    # ETAPA 1: Preencher todos os triângulos (scan-line)
    for tri in mesh.faces:
        verts = [(float(x), float(y)) for x, y in v2d_clamped[list(tri)]]
        for x, y in _safe_scanline(verts):
            if 0 <= x < width and 0 <= y < height:
                img[int(y), int(x)] = 255

    # ETAPA 2: Desenhar as arestas por cima para cobrir as frestas
    drawn_edges = set() # Usar um 'set' é crucial para a performance
    for tri in mesh.faces:
        idx = list(tri)
        for i in range(3):
            p1 = tuple(v2d_int[idx[i]])
            p2 = tuple(v2d_int[idx[(i + 1) % 3]])

            # Cria uma representação única da aresta para evitar duplicatas
            edge = tuple(sorted((p1, p2)))

            if edge not in drawn_edges:
                _draw_line(img, p1, p2)
                drawn_edges.add(edge)


# ------------------------------------------------------------------ #
#  Rasterização da cena inteira                                      #
# ------------------------------------------------------------------ #
def rasterize_scene(meshes: List[Mesh],
                    M_vp: np.ndarray,
                    width: int = 400,
                    height: int = 400) -> np.ndarray:
    """
    Cria frame-buffer H×W uint8 combinando todos os meshes,
    usando _scan-line_ para sólidos e Bresenham para linhas.
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for mesh in meshes:
        _rasterize_single(mesh, M_vp, width, height, img)
    return img