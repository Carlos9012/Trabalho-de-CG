"""
meshing.py
Builds triangle faces from the vertex layouts produced by geometry.py.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _add_quad(faces: list, a: int, b: int, c: int, d: int):
    """Add two triangles for the quad (a, b, c, d)."""
    faces.append([a, b, c])
    faces.append([a, c, d])

# ---------------------------------------------------------------------
# primitive face generators
# ---------------------------------------------------------------------

def _faces_box() -> np.ndarray:
    """Faces for the unit box created by create_box()."""
    return np.array([
        [0, 1, 2], [0, 2, 3],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 4, 5], [0, 5, 1],   # -x
        [1, 5, 6], [1, 6, 2],   # +y
        [2, 6, 7], [2, 7, 3],   # +x
        [3, 7, 4], [3, 4, 0],   # -y
    ], dtype=int)

def _faces_cylinder(v: np.ndarray, radial: int, hseg: int, cap: bool) -> np.ndarray:
    """Faces for a capped right cylinder."""
    faces = []
    rings = hseg + 1

    # side wall
    for r in range(hseg):
        base = r * radial
        nxt = (r + 1) * radial
        for i in range(radial):
            a, b = base + i, base + (i + 1) % radial
            c, d = nxt + (i + 1) % radial, nxt + i
            _add_quad(faces, a, b, c, d)

    if cap:
        top_center = len(v) - 1
        bot_center = len(v) - 2
        top_start = (rings - 1) * radial
        for i in range(radial):
            faces.append([top_center, top_start + i, top_start + (i + 1) % radial])
            faces.append([bot_center, (i + 1) % radial, i])
    return np.array(faces, dtype=int)

def _faces_pipe_tube(radial: int, rings: int) -> np.ndarray:
    """Faces for a single‑wall tube (helper, unused for double tubes)."""
    faces = []
    for r in range(rings - 1):
        base = r * radial
        nxt = (r + 1) * radial
        for i in range(radial):
            a, b = base + i, base + (i + 1) % radial
            c, d = nxt + (i + 1) % radial, nxt + i
            _add_quad(faces, a, b, c, d)
    return np.array(faces, dtype=int)

def _faces_double_tube(radial: int, rings: int, cap: bool = True) -> np.ndarray:
    """Faces for a hollow tube with constant thickness."""
    faces: list[list[int]] = []

    # outer and inner walls
    for r in range(rings - 1):
        o0 = (2 * r) * radial
        i0 = o0 + radial
        o1 = o0 + 2 * radial
        i1 = i0 + 2 * radial
        for k in range(radial):
            k1 = (k + 1) % radial
            _add_quad(faces, o0 + k, o0 + k1, o1 + k1, o1 + k)  # outer
            _add_quad(faces, i0 + k, i1 + k, i1 + k1, i0 + k1)  # inner

    # radial band between walls
    for r in range(rings):
        o = (2 * r) * radial
        i = o + radial
        for k in range(radial):
            k1 = (k + 1) % radial
            _add_quad(faces, o + k, o + k1, i + k1, i + k)

    # optional caps
    if cap:
        o0, i0 = 0, radial
        oL = (2 * (rings - 1)) * radial
        iL = oL + radial
        for k in range(radial):
            k1 = (k + 1) % radial
            _add_quad(faces, o0 + k, i0 + k, i0 + k1, o0 + k1)  # bottom
            _add_quad(faces, oL + k, iL + k, iL + k1, oL + k1)  # top

    return np.array(faces, dtype=int)

# ---------------------------------------------------------------------
# public api
# ---------------------------------------------------------------------

def generate_mesh(name: str, vertices: np.ndarray, edges, **kwargs):
    """Return (vertices, faces) for a primitive identified by *name*."""
    v = vertices

    if name == "box":
        f = _faces_box()

    elif name == "cylinder":
        f = _faces_cylinder(
            v,
            kwargs.get("radial_segments", 32),
            kwargs.get("height_segments", 1),
            kwargs.get("cap_ends", True),
        )

    elif name in ("pipe_straight", "pipe_curved"):
        if "radial_segments" not in kwargs:
            raise ValueError("radial_segments is required for tubes.")
        radial = kwargs["radial_segments"]
        rings = len(v) // (2 * radial)
        f = _faces_double_tube(radial, rings, kwargs.get("cap_ends", True))

    else:
        f = np.empty((0, 3), int)

    return v, f
