"""
geometry.py
Modelagem de sólidos retornando **apenas vertices e edges**.

Primitivas disponíveis
----------------------
create_line           – reta de tamanho 4 (default)
create_cylinder       – cilindro (tampado ou oco)
create_pipe_straight  – cano reto (cilindro oco)
create_box            – paralelepípedo (hiper-retângulo)
create_pipe_curved    – cano curvo via Hermite

Cada função retorna:
    vertices : np.ndarray  (N, 3)
    edges    : list[tuple[int, int]]
              (arestas únicas, índices ordenados)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List
from algebra import hermite_curve, normalize

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _add_edge(edge_set: set, a: int, b: int):
    """Adiciona aresta (a,b) ao set, já ordenada para evitar duplicatas."""
    edge_set.add((a, b) if a < b else (b, a))

def _ring_vertices(radius, angle):
    """Retorna coordenadas (x,y) de um anel no plano XY."""
    return np.column_stack([np.cos(angle)*radius, np.sin(angle)*radius])

def _frenet_frame(tangent: np.ndarray):
    tangent = normalize(tangent)
    ref     = np.array([0,0,1]) if abs(tangent[2]) < .9 else np.array([0,1,0])
    normal  = normalize(np.cross(tangent, ref))
    binorm  = np.cross(tangent, normal)
    return normal, binorm

# ---------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------

def create_line(
    length: float = 4.0,
    origin: Tuple[float, float, float] = (0,0,0),
    direction: Tuple[float, float, float] = (1,0,0),
):
    """Reta simples com 2 vértices."""
    origin    = np.asarray(origin, float)
    direction = normalize(direction)
    vertices  = np.stack([origin, origin + direction*length])
    edges     = [(0,1)]
    return vertices, edges

# ---------- Cilindro / Cano reto ---------- #

def create_cylinder(
    radius: float,
    height: float,
    radial_segments: int = 32,
    height_segments: int = 1,
    origin: Tuple[float,float,float] = (0,0,0),
    cap_ends: bool = True,
):
    """
    Cilindro sólido (cap_ends=True) ou oco (False).
    Retorna apenas vertices e edges (sem faces).
    """
    ox, oy, oz = origin
    angle      = np.linspace(0, 2*np.pi, radial_segments, endpoint=False)
    z_vals     = np.linspace(0, height, height_segments+1)

    # Gera vértices anel-a-anel
    verts_xy   = _ring_vertices(radius, angle)          # (radial,2)
    vertices   = []
    for z in z_vals:
        ring = np.column_stack([verts_xy,
                                np.full(radial_segments, oz+z)])
        ring[:,0] += ox
        ring[:,1] += oy
        vertices.append(ring)
    vertices = np.vstack(vertices)                      # ((h+1)*radial,3)

    edge_set: set[tuple[int,int]] = set()
    rings = height_segments+1

    # Arestas ao longo da circunferência
    for r in range(rings):
        base = r*radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, base+i,
                                base+((i+1)%radial_segments))

    # Arestas verticais
    for r in range(rings-1):
        base, nxt = r*radial_segments, (r+1)*radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, base+i, nxt+i)

    # Tampas (opcional) – arestas entre centro e anel
    if cap_ends:
        # fundo
        center_bot = len(vertices)
        vertices = np.vstack([vertices, [ox, oy, oz]])
        for i in range(radial_segments):
            _add_edge(edge_set, center_bot, i)
        # topo
        center_top = len(vertices)
        vertices = np.vstack([vertices, [ox, oy, oz+height]])
        top_base  = (rings-1)*radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, center_top, top_base+i)

    edges = sorted(edge_set)
    return vertices, edges

def create_pipe_straight(
    radius: float,
    length: float,
    radial_segments: int = 32,
    length_segments: int = 1,
    origin: Tuple[float,float,float] = (0,0,0),
):
    """Cano reto (cilindro sem tampas)."""
    return create_cylinder(
        radius, length, radial_segments,
        length_segments, origin, cap_ends=False
    )

# ---------- Paralelepípedo ---------- #

def create_box(
    width: float,
    depth: float,
    height: float,
    origin: Tuple[float,float,float] = (0,0,0),
):
    """Caixa eixo-alinhada."""
    ox, oy, oz = origin
    # 8 vértices
    vertices = np.array(
        [[ox,         oy,          oz         ],  # 0
         [ox+width,   oy,          oz         ],  # 1
         [ox+width,   oy+depth,    oz         ],  # 2
         [ox,         oy+depth,    oz         ],  # 3
         [ox,         oy,          oz+height ],  # 4
         [ox+width,   oy,          oz+height ],  # 5
         [ox+width,   oy+depth,    oz+height ],  # 6
         [ox,         oy+depth,    oz+height ]], # 7
        float,
    )

    # 12 arestas
    edges = [(0,1),(1,2),(2,3),(3,0),  # base
             (4,5),(5,6),(6,7),(7,4),  # topo
             (0,4),(1,5),(2,6),(3,7)]  # colunas
    return vertices, edges

# ---------- Cano curvo ---------- #

def create_pipe_curved(
    radius: float,
    p0: Tuple[float,float,float],
    p1: Tuple[float,float,float],
    t0: Tuple[float,float,float],
    t1: Tuple[float,float,float],
    curve_samples: int = 50,
    radial_segments: int = 16,
):
    """Cano que segue uma curva Hermite cúbica."""
    t_vals = np.linspace(0,1,curve_samples)
    path   = hermite_curve(np.asarray(p0), np.asarray(p1),
                           np.asarray(t0), np.asarray(t1), t_vals)
    tangents = np.gradient(path, axis=0)

    angle = np.linspace(0, 2*np.pi, radial_segments, endpoint=False)
    edge_set: set[tuple[int,int]] = set()
    vertices = []

    for j,(pt,tan) in enumerate(zip(path, tangents)):
        n,b = _frenet_frame(tan)
        ring = pt + np.outer(np.cos(angle), n*radius) \
                 + np.outer(np.sin(angle), b*radius)
        base = len(vertices)
        vertices.extend(ring)

        # Arestas circumferenciais
        for i in range(radial_segments):
            _add_edge(edge_set, base+i, base+((i+1)%radial_segments))

        # Arestas longitudinais com anel anterior
        if j > 0:
            prev_base = base - radial_segments
            for i in range(radial_segments):
                _add_edge(edge_set, prev_base+i, base+i)

    vertices = np.asarray(vertices, float)
    edges    = sorted(edge_set)
    return vertices, edges
