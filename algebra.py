"""algebra.py
Linear‑algebra helpers for geometry.py.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------
# Hermite spline
# ---------------------------------------------------------------------
def hermite_basis() -> np.ndarray:
    """4×4 cubic Hermite basis."""
    return np.array(
        [[2, -2, 1, 1],
         [-3, 3, -2, -1],
         [0, 0, 1, 0],
         [1, 0, 0, 0]],
        float,
    )


def hermite_curve(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluates the Hermite curve for parameters *t* ∈ [0, 1]."""
    H = hermite_basis()
    C = H @ np.stack([p0, p1, t0, t1])
    T = np.stack([t**3, t**2, t, np.ones_like(t)], axis=-1)
    return T @ C


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Returns *v* normalized."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero‑length vector.")
    return v / n


# ---------------------------------------------------------------------
# 4×4 homogeneous transforms
# ---------------------------------------------------------------------
def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.diag([sx, sy, sz, 1.0])


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0, 0, 1]])


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


# ---------------------------------------------------------------------
# Apply transform
# ---------------------------------------------------------------------
def apply_transform(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
    v_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    v_new = (M @ v_h.T).T
    return v_new[:, :3] / v_new[:, 3:]


# ---------------------------------------------------------------------
# Camera (R · T)
# ---------------------------------------------------------------------
def camera_R_T(eye, at, up=np.array([0, 1, 0], float)):
    eye, at, up = map(np.asarray, (eye, at, up))
    n = normalize(eye - at)
    u = normalize(np.cross(up, n))
    v = np.cross(n, u)
    R = np.array([[*u, 0],
                  [*v, 0],
                  [*n, 0],
                  [0, 0, 0, 1]], float)
    T = np.eye(4, dtype=float)
    T[:3, 3] = -eye
    V = R @ T
    return R, T, V


# ---------------------------------------------------------------------
# Perspective projection
# ---------------------------------------------------------------------
def perspective_matrix(fov_y: float, aspect: float,
                       near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y) / 2)
    nf = 1 / (near - far)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) * nf, 2 * far * near * nf],
        [0, 0, -1, 0]
    ], float)


def project_ndc_to_screen(ndc_xy: np.ndarray,
                          width: int, height: int) -> np.ndarray:
    x_ndc, y_ndc = ndc_xy.T
    x_scr = (x_ndc + 1) * 0.5 * width
    y_scr = (1 - y_ndc) * 0.5 * height
    return np.vstack([x_scr, y_scr]).T
