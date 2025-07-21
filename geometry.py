"""
geometry.py – Sólidos/objetos exigidos no trabalho.

• Retorna Mesh(vertices, faces) com índices 0-based.
• Usa apenas numpy + math; sem libs externas.
"""

from __future__ import annotations
import numpy as np, math
from dataclasses import dataclass
import utils2 as ut                    # para curva Hermite

# ------------------------------------------------------------------ #
#  Estrutura base                                                    #
# ------------------------------------------------------------------ #
@dataclass
class Mesh:
    vertices: np.ndarray   # (N,3)
    faces:    np.ndarray   # (M,3)  — vazio se é linha

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())


# ------------------------------------------------------------------ #
#  Objetos simples já existentes                                     #
# ------------------------------------------------------------------ #
def box(lx=2.0, ly=1.0, lz=1.0) -> Mesh:
    """Paralelepípedo com 8 vértices, 12 triângulos."""
    x, y, z = lx / 2, ly / 2, lz / 2
    v = np.array([(-x,-y,-z),( x,-y,-z),( x, y,-z),(-x, y,-z),
                  (-x,-y, z),( x,-y, z),( x, y, z),(-x, y, z)])
    f = np.array([(0,1,2),(0,2,3),(4,6,5),(4,7,6),
                  (0,4,5),(0,5,1),(1,5,6),(1,6,2),
                  (2,6,7),(2,7,3),(3,7,4),(3,4,0)])
    return Mesh(v,f)


def cylinder(radius=1.0, height=4.0,
             n_segments=20, n_layers=1) -> Mesh:
    """Cilindro fechado (tampas + laterais)."""
    verts, faces = [], []
    # anéis
    for k in range(n_layers+1):
        z = height*k/n_layers
        for i in range(n_segments):
            t = 2*math.pi*i/n_segments
            verts.append((radius*math.cos(t), radius*math.sin(t), z))
    # tampas (centros)
    idx_base, idx_top = len(verts), len(verts)+1
    verts += [(0,0,0),(0,0,height)]

    for k in range(n_layers):
        o0, o1 = k*n_segments, (k+1)*n_segments
        for i in range(n_segments):
            j = (i+1)%n_segments
            a,b,c,d = o0+i, o0+j, o1+j, o1+i
            faces += [(a,b,c),(a,c,d)]               # lateral
            if k==0:   faces.append((idx_base,b,a))  # tampa base
            if k==n_layers-1: faces.append((idx_top,d,c))  # tampa topo
    return Mesh(np.asarray(verts,float), np.asarray(faces,int))


def line(p0:tuple[float,float,float],
         p1:tuple[float,float,float]) -> Mesh:
    """Linha (dois vértices, sem faces)."""
    return Mesh(np.array([p0,p1],float), np.empty((0,3),int))


# ------------------------------------------------------------------ #
#  Cano reto  (≈ cilindro “oco”)                                     #
# ------------------------------------------------------------------ #
def pipe_straight(radius=1.0, length=4.0,
                  thickness=0.2,
                  n_segments=20) -> Mesh:
    """
    Gera cano reto com parede de ‘thickness’.
    • Duas superfícies: externa (raio R) e interna (R-th).
    """
    R_out, R_in = radius, max(radius-thickness, 1e-3)
    verts, faces = [], []
    # anéis externos + internos (0-segms = z0, segms = z1)
    for r in (R_out, R_in):
        for z in (0.0, length):
            for i in range(n_segments):
                t = 2*math.pi*i/n_segments
                verts.append((r*math.cos(t), r*math.sin(t), z))
    # índices auxiliares
    ext0 = 0
    ext1 = n_segments
    int0 = 2*n_segments
    int1 = 3*n_segments
    # laterais externa + interna
    for ring_a, ring_b in ((ext0,ext1),(int1,int0)):  # obs ordem p/face interna invertida
        for i in range(n_segments):
            j=(i+1)%n_segments
            a,b,c,d = ring_a+i, ring_a+j, ring_b+j, ring_b+i
            faces += [(a,b,c),(a,c,d)]
    # tampas (une ext -> int)
    for ring_e, ring_i in ((ext0,int0),(ext1,int1)):
        for i in range(n_segments):
            j=(i+1)%n_segments
            faces += [(ring_e+i, ring_i+i, ring_i+j),
                      (ring_e+i, ring_i+j, ring_e+j)]
    return Mesh(np.asarray(verts,float), np.asarray(faces,int))


# ------------------------------------------------------------------ #
#  Cano curvo  (extrusão em curva Hermite)                           #
# ------------------------------------------------------------------ #
def pipe_curved(radius=1.0,
                p1=(0,0,0), p2=(4,4,0),
                t1=(4,0,0), t2=(0,4,0),
                n_curve=24, n_circle=16) -> Mesh:
    """
    Extrusa um círculo de raio ‘radius’ ao longo de uma curva de Hermite.
    • p1,p2 – pontos; t1,t2 – tangentes (mesma convenção do prof.).
    """
    # 1. pontos da curva
    C = ut.hermite(np.array(p1), np.array(p2),
                   np.array(t1), np.array(t2),
                   num_pontos=n_curve)
    verts, faces = [], []
    up_ref = np.array([0,0,1])

    def frame(i:int):
        """Retorna vetores (t,u,v) ortonormais para anel i."""
        if i==len(C)-1: a=C[i]-C[i-1]
        else:           a=C[i+1]-C[i]
        t = ut.normalizar_vetor(a)
        # evita colinearidade escolhendo outro vetor de referência
        if abs(np.dot(t, up_ref))>0.9:
            up = np.array([0,1,0])
        else: up = up_ref
        v = ut.normalizar_vetor(np.cross(t, up))
        u = np.cross(v, t)
        return t,u,v

    # 2. gera anéis
    for i,center in enumerate(C):
        _,u,v = frame(i)
        for k in range(n_circle):
            ang = 2*math.pi*k/n_circle
            pt = center + radius*math.cos(ang)*u + radius*math.sin(ang)*v
            verts.append(tuple(pt))

    # 3. faces entre anéis
    for i in range(n_curve-1):
        off0, off1 = i*n_circle, (i+1)*n_circle
        for k in range(n_circle):
            j=(k+1)%n_circle
            a,b,c,d = off0+k, off0+j, off1+j, off1+k
            faces += [(a,b,c),(a,c,d)]
    # 4. tampas (opcional: não exigido no enunciado)
    return Mesh(np.asarray(verts,float), np.asarray(faces,int))
