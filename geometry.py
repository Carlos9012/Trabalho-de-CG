"""geometry.py
Vertices and edges for basic solids.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple
from algebra import hermite_curve, normalize


def _add_edge(edge_set: set, a: int, b: int):
    edge_set.add((a, b) if a < b else (b, a))


def _ring_vertices(radius: float, angle: np.ndarray) -> np.ndarray:
    return np.column_stack([np.cos(angle) * radius, np.sin(angle) * radius])


def _frenet_frame(tangent: np.ndarray):
    tangent = normalize(tangent)
    ref = np.array([0, 0, 1]) if abs(tangent[2]) < 0.9 else np.array([0, 1, 0])
    normal = normalize(np.cross(tangent, ref))
    binorm = np.cross(tangent, normal)
    return normal, binorm


def create_line(
    length: float = 4.0,
    origin: Tuple[float, float, float] = (0, 0, 0),
    direction: Tuple[float, float, float] = (1, 0, 0),
):
    """Straight line with two vertices."""
    origin = np.asarray(origin, float)
    direction = normalize(direction)
    vertices = np.stack([origin, origin + direction * length])
    edges = [(0, 1)]
    return vertices, edges


def create_cylinder(
    radius: float,
    height: float,
    radial_segments: int = 32,
    height_segments: int = 1,
    origin: Tuple[float, float, float] = (0, 0, 0),
    cap_ends: bool = True,
):
    """Cylinder, solid or hollow, returning vertices and edges."""
    ox, oy, oz = origin
    angle = np.linspace(0, 2 * np.pi, radial_segments, endpoint=False)
    z_vals = np.linspace(0, height, height_segments + 1)
    verts_xy = _ring_vertices(radius, angle)
    vertices = []
    for z in z_vals:
        ring = np.column_stack([verts_xy, np.full(radial_segments, oz + z)])
        ring[:, 0] += ox
        ring[:, 1] += oy
        vertices.append(ring)
    vertices = np.vstack(vertices)
    edge_set: set[tuple[int, int]] = set()
    rings = height_segments + 1
    for r in range(rings):
        base = r * radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, base + i, base + (i + 1) % radial_segments)
    for r in range(rings - 1):
        base, nxt = r * radial_segments, (r + 1) * radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, base + i, nxt + i)
    if cap_ends:
        center_bot = len(vertices)
        vertices = np.vstack([vertices, [ox, oy, oz]])
        for i in range(radial_segments):
            _add_edge(edge_set, center_bot, i)
        center_top = len(vertices)
        vertices = np.vstack([vertices, [ox, oy, oz + height]])
        top_base = (rings - 1) * radial_segments
        for i in range(radial_segments):
            _add_edge(edge_set, center_top, top_base + i)
    return np.asarray(vertices, float), sorted(edge_set)


def create_pipe_straight(
    r_inner: float,
    r_outer: float,
    length: float,
    radial_segments: int = 32,
    length_segments: int = 1,
    origin: Tuple[float, float, float] = (0, 0, 0),
):
    """Straight hollow tube with thickness."""
    if r_outer <= r_inner or r_inner <= 0:
        raise ValueError("Require 0 < r_inner < r_outer.")
    ox, oy, oz = origin
    angle = np.linspace(0, 2 * np.pi, radial_segments, endpoint=False)
    z_vals = np.linspace(0, length, length_segments + 1)
    outer_xy = _ring_vertices(r_outer, angle)
    inner_xy = _ring_vertices(r_inner, angle)
    vertices = []
    edge_set: set[tuple[int, int]] = set()
    for k, z in enumerate(z_vals):
        zcoord = np.full(radial_segments, oz + z)
        ring_o = np.column_stack([outer_xy, zcoord])
        ring_o[:, 0] += ox
        ring_o[:, 1] += oy
        base_o = len(vertices)
        vertices.extend(ring_o)
        ring_i = np.column_stack([inner_xy, zcoord])
        ring_i[:, 0] += ox
        ring_i[:, 1] += oy
        base_i = len(vertices)
        vertices.extend(ring_i)
        for i in range(radial_segments):
            _add_edge(edge_set, base_o + i, base_o + (i + 1) % radial_segments)
            _add_edge(edge_set, base_i + i, base_i + (i + 1) % radial_segments)
        for i in range(radial_segments):
            _add_edge(edge_set, base_o + i, base_i + i)
        if k > 0:
            prev_o = base_o - 2 * radial_segments
            prev_i = base_i - 2 * radial_segments
            for i in range(radial_segments):
                _add_edge(edge_set, prev_o + i, base_o + i)
                _add_edge(edge_set, prev_i + i, base_i + i)
    return np.asarray(vertices, float), sorted(edge_set)


def create_box(
    width: float,
    depth: float,
    height: float,
    origin: Tuple[float, float, float] = (0, 0, 0),
):
    """Axis‑aligned box."""
    ox, oy, oz = origin
    vertices = np.array(
        [
            [ox, ox + 0, oz],
            [ox + width, oy, oz],
            [ox + width, oy + depth, oz],
            [ox, oy + depth, oz],
            [ox, oy, oz + height],
            [ox + width, oy, oz + height],
            [ox + width, oy + depth, oz + height],
            [ox, oy + depth, oz + height],
        ],
        float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return vertices, edges


def create_pipe_curved(
    r_inner: float,
    r_outer: float,
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    t0: Tuple[float, float, float],
    t1: Tuple[float, float, float],
    curve_samples: int = 50,
    radial_segments: int = 16,
):
    """Curved hollow tube following a Hermite path."""
    if r_outer <= r_inner or r_inner <= 0:
        raise ValueError("Require 0 < r_inner < r_outer.")
    angle = np.linspace(0, 2 * np.pi, radial_segments, endpoint=False)
    path = hermite_curve(np.asarray(p0), np.asarray(p1), np.asarray(t0), np.asarray(t1), np.linspace(0, 1, curve_samples))
    tangents = np.gradient(path, axis=0)
    vertices = []
    edge_set: set[tuple[int, int]] = set()
    for j, (pt, tan) in enumerate(zip(path, tangents)):
        n, b = _frenet_frame(tan)
        ring_o = pt + np.outer(np.cos(angle), n * r_outer) + np.outer(np.sin(angle), b * r_outer)
        base_o = len(vertices)
        vertices.extend(ring_o)
        ring_i = pt + np.outer(np.cos(angle), n * r_inner) + np.outer(np.sin(angle), b * r_inner)
        base_i = len(vertices)
        vertices.extend(ring_i)
        for i in range(radial_segments):
            _add_edge(edge_set, base_o + i, base_o + (i + 1) % radial_segments)
            _add_edge(edge_set, base_i + i, base_i + (i + 1) % radial_segments)
        for i in range(radial_segments):
            _add_edge(edge_set, base_o + i, base_i + i)
        if j > 0:
            prev_o = base_o - 2 * radial_segments
            prev_i = base_i - 2 * radial_segments
            for i in range(radial_segments):
                _add_edge(edge_set, prev_o + i, base_o + i)
                _add_edge(edge_set, prev_i + i, base_i + i)
    return np.asarray(vertices, float), sorted(edge_set)
