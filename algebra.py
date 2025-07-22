"""
algebra.py
Utilidades de álgebra linear usadas por geometry.py.

Mantém em um só lugar as fórmulas analíticas (ex.: spline de Hermite),
deixando geometry.py focado apenas na geração de sólidos.
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------
# Hermite spline
# ----------------------------------------------------------------------
def hermite_basis() -> np.ndarray:
    """Retorna a matriz base 4×4 do polinômio cúbico de Hermite."""
    return np.array(
        [[ 2, -2,  1,  1],
         [-3,  3, -2, -1],
         [ 0,  0,  1,  0],
         [ 1,  0,  0,  0]],
        dtype=float,
    )


def hermite_curve(
    p0: np.ndarray,
    p1: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Avalia uma curva cúbica de Hermite nos parâmetros *t* (0‥1).

    Parameters
    ----------
    p0, p1 : (3,) array_like
        Posições inicial e final.
    t0, t1 : (3,) array_like
        Tangentes em *p0* e *p1*.
    t : array_like
        Escalar ou array de parâmetros onde a curva será avaliada.

    Returns
    -------
    pts : (len(t), 3) ndarray
        Pontos da curva.
    """
    H = hermite_basis()            # (4,4)
    G = np.stack([p0, p1, t0, t1]) # (4,3)
    C = H @ G                      # (4,3)

    T = np.stack([t**3, t**2, t, np.ones_like(t)], axis=-1)  # (len(t),4)
    return T @ C                   # (len(t),3)


# ----------------------------------------------------------------------
# Utilidades gerais
# ----------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """Normaliza e devolve o vetor *v*."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Vetor de comprimento zero não pode ser normalizado.")
    return v / n



# ----------------------------------------------------------------------
# Matrizes de transformação homogênea 4×4
# ----------------------------------------------------------------------
def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    """Matriz de translação homogênea."""
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M

def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    """Matriz de escala homogênea."""
    return np.diag([sx, sy, sz, 1.0])

def rot_x(theta: float) -> np.ndarray:
    """Rotação em torno do eixo X (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rot_y(theta: float) -> np.ndarray:
    """Rotação em torno do eixo Y (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rot_z(theta: float) -> np.ndarray:
    """Rotação em torno do eixo Z (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# ----------------------------------------------------------------------
# Aplicação de transformações
# ----------------------------------------------------------------------
def apply_transform(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Aplica matriz homogênea 4×4 sobre vértices (N×3) e devolve novos vértices.
    """
    v_h    = np.hstack([vertices, np.ones((len(vertices),1))])  # (N,4)
    v_new  = (M @ v_h.T).T                                      # (N,4)
    return v_new[:, :3] / v_new[:, 3:]









# ----------------------------------------------------------------------
# Câmera  (R · T · p_w)   — linhas:  u  v  n   ;   n = eye − at
# ----------------------------------------------------------------------
def camera_R_T(eye, at, up=np.array([0, 1, 0], float)):
    """
    Retorna (R, T, V) onde:
        • R – rotação (linhas = u, v, n)
        • T – translação (−eye)
        • V = R @ T (matriz-view completa)

    eye, at, up : iteráveis de 3 floats
    """
    eye, at, up = map(np.asarray, (eye, at, up))

    n = normalize(eye - at)                 # eixo z (para trás)
    u = normalize(np.cross(up, n))          # eixo x
    v = np.cross(n, u)                      # eixo y

    R = np.array([
        [*u, 0],
        [*v, 0],
        [*n, 0],
        [ 0, 0, 0, 1]
    ], float)

    T = np.eye(4, dtype=float)
    T[:3, 3] = -eye

    V = R @ T                               # p_c = V · p_w
    return R, T, V






# ----------------------------------------------------------------------
# Projeção em perspectiva  (Questão 4)
# ----------------------------------------------------------------------
def perspective_matrix(fov_y: float, aspect: float,
                       near: float, far: float) -> np.ndarray:
    """
    Retorna a matriz 4×4 de projeção perspectiva (RH, NDC ∈ [-1,1]).

    Parameters
    ----------
    fov_y  : float  – campo de visão vertical (graus)
    aspect : float  – largura/altura da janela
    near, far : float
    """
    f = 1.0 / np.tan(np.deg2rad(fov_y) / 2)
    nf = 1 / (near - far)
    return np.array([
        [f/aspect, 0, 0,                     0],
        [0,        f, 0,                     0],
        [0,        0, (far+near)*nf, 2*far*near*nf],
        [0,        0, -1,                    0]
    ], dtype=float)

def project_ndc_to_screen(ndc_xy: np.ndarray,
                          width: int, height: int) -> np.ndarray:
    """
    Converte coordenadas NDC (-1..1) para pixels (0..width-1, 0..height-1).

    Retorna (N,2) float.
    """
    x_ndc, y_ndc = ndc_xy.T
    x_scr = (x_ndc + 1) * 0.5 * width
    y_scr = (1 - y_ndc) * 0.5 * height   # origem no canto superior-esquerdo
    return np.vstack([x_scr, y_scr]).T
