"""
meshing.py
Gera faces (triângulos) a partir dos vértices que já vêm de geometry.py,
sem usar marching-cubes ou casco convexo.

Regras por objeto:
* box            – 12 triângulos (6 faces × 2)
* cylinder       – tampado (topo+base)
* pipe_straight  – tubo aberto (sem tampas)
* pipe_curved    – tubo curvo, aberto
* outros         – retorna faces vazias (para futura extensão)

A rotina usa o parâmetro `name` passado pelo chamador para descobrir a
topologia esperada.
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------
# Utilidades gerais
# ----------------------------------------------------------------------
def _add_quad(faces: list, a: int, b: int, c: int, d: int):
    """Adiciona os dois triângulos do quadrilátero (a,b,c,d)."""
    faces.append([a, b, c])
    faces.append([a, c, d])

# ----------------------------------------------------------------------
# Funções específicas
# ----------------------------------------------------------------------
def _faces_box() -> np.ndarray:
    """Retorna as 12 faces padrão da caixa ‘create_box’."""
    return np.array([
        [0,1,2],[0,2,3],   # base (z = 0)
        [4,5,6],[4,6,7],   # topo
        [0,4,5],[0,5,1],   # lado -x
        [1,5,6],[1,6,2],   # lado +y
        [2,6,7],[2,7,3],   # lado +x
        [3,7,4],[3,4,0],   # lado -y
    ], int)

def _faces_cylinder(v: np.ndarray, radial: int, height_segments: int,
                    cap_ends: bool) -> np.ndarray:
    faces = []
    rings = height_segments + 1

    # faces laterais
    for r in range(height_segments):
        base = r * radial
        nxt  = (r+1) * radial
        for i in range(radial):
            a = base + i
            b = base + (i+1)%radial
            c = nxt  + (i+1)%radial
            d = nxt  + i
            _add_quad(faces, a, b, c, d)

    if cap_ends:
        # topo (z maior)
        center_top = len(v) - 1
        center_bot = len(v) - 2
        top_ring   = (rings-1) * radial
        for i in range(radial):
            faces.append([center_top, top_ring+i, top_ring+(i+1)%radial])
        # fundo
        for i in range(radial):
            faces.append([center_bot, i, (i+1)%radial])
    return np.array(faces, int)

def _faces_pipe_tube(radial: int, rings: int) -> np.ndarray:
    """Triangulação padrão para qualquer tubo (reto ou curvo)."""
    faces = []
    for r in range(rings-1):
        base = r     * radial
        nxt  = (r+1) * radial
        for i in range(radial):
            a = base + i
            b = base + (i+1)%radial
            c = nxt  + (i+1)%radial
            d = nxt  + i
            _add_quad(faces, a, b, c, d)
    return np.array(faces, int)

# ----------------------------------------------------------------------
# Função pública
# ----------------------------------------------------------------------
def generate_mesh(name: str,
                  vertices: np.ndarray,
                  edges,                       # ignorado (compat.)
                  **kwargs):
    """
    Parameters
    ----------
    name      : str   – identificador da primitiva (mesmo usado em example.py)
    vertices  : (N,3) ndarray
    edges     : não usado
    kwargs    : parâmetros específicos (opcionais)

    Returns
    -------
    vertices : (N,3) ndarray   (inalterados)
    faces    : (K,3) ndarray[int]  – triângulos
    """

    v = vertices
    faces: np.ndarray

    if name == "box":
        faces = _faces_box()

    elif name == "cylinder":
        radial = kwargs.get("radial_segments", 32)
        hseg   = kwargs.get("height_segments", 1)
        cap    = kwargs.get("cap_ends", True)
        faces  = _faces_cylinder(v, radial, hseg, cap)

    elif name == "pipe_straight":
        radial = kwargs.get("radial_segments", 32)
        rings  = kwargs.get("length_segments", 1) + 1
        faces  = _faces_pipe_tube(radial, rings)

    elif name == "pipe_curved":
        radial = kwargs.get("radial_segments", 16)
        rings  = len(v) // radial
        faces  = _faces_pipe_tube(radial, rings)

    else:
        # fallback: sem malha
        faces = np.empty((0,3), int)

    return v, faces
