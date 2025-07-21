import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def criar_view_matrix_classica(eye, at, up):
    eye = np.array(eye)
    at = np.array(at)
    up = np.array(up)

    n = eye - at
    n = n / np.linalg.norm(n)
    u = np.cross(up, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    M = np.identity(4)
    M[:3, :3] = np.vstack([u, v, n])
    T = np.identity(4)
    T[:3, 3] = -eye
    return M @ T, u, v, n

def criar_model_matrix(scale=1.0, rotation_axis=(0,1,0), rotation_angle=0, translation=(0,0,0)):
    S = np.eye(4)
    S[0,0] = S[1,1] = S[2,2] = scale
    angle = np.radians(rotation_angle)
    x, y, z = rotation_axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c+(1-c)*x*x, (1-c)*x*y - s*z, (1-c)*x*z + s*y, 0],
        [(1-c)*y*x + s*z, c+(1-c)*y*y, (1-c)*y*z - s*x, 0],
        [(1-c)*z*x - s*y, (1-c)*z*y + s*x, c+(1-c)*z*z, 0],
        [0, 0, 0, 1]
    ])
    T = np.eye(4)
    T[:3, 3] = translation
    return T @ R @ S

def aplicar_transformacao(vertices, M):
    v = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    v = (M @ v.T).T
    return v[:, :3] / v[:, [3]]

# --- VISUALIZAÇÃO ---
def desenhar_camera(ax, eye, at, fov, aspect, near, far, u, v, n):
    h_near = 2 * np.tan(np.radians(fov / 2)) * near
    w_near = h_near * aspect
    h_far = 2 * np.tan(np.radians(fov / 2)) * far
    w_far = h_far * aspect
    nc = eye - n * near
    fc = eye - n * far

    near_tl = nc + v * h_near/2 - u * w_near/2
    near_tr = nc + v * h_near/2 + u * w_near/2
    near_bl = nc - v * h_near/2 - u * w_near/2
    near_br = nc - v * h_near/2 + u * w_near/2
    far_tl = fc + v * h_far/2 - u * w_far/2
    far_tr = fc + v * h_far/2 + u * w_far/2
    far_bl = fc - v * h_far/2 - u * w_far/2
    far_br = fc - v * h_far/2 + u * w_far/2

    linhas = [
        [eye, near_tl], [eye, near_tr], [eye, near_bl], [eye, near_br],
        [near_tl, near_tr], [near_tr, near_br], [near_br, near_bl], [near_bl, near_tl],
        [far_tl, far_tr], [far_tr, far_br], [far_br, far_bl], [far_bl, far_tl],
        [near_tl, far_tl], [near_tr, far_tr], [near_bl, far_bl], [near_br, far_br]
    ]

    for linha in linhas:
        x, y, z = zip(*linha)
        ax.plot(x, y, z, color='black', linewidth=0.5)
    for linha in [[near_tl, near_tr], [near_tr, near_br], [near_br, near_bl], [near_bl, near_tl]]:
        x, y, z = zip(*linha)
        ax.plot(x, y, z, color='red', linewidth=2.5)

    ax.scatter(0, 0, 0, color='red', s=60)
    ax.text(0, 0, 0, '  Origem (0,0,0)', color='red', fontsize=10)

    scale = 1.5
    ax.quiver(*eye, *(u * scale), color='blue', label='u (Right)')
    ax.quiver(*eye, *(v * scale), color='green', label='v (Up)')
    ax.quiver(*eye, *(-n * scale), color='orange', label='-n (Forward)')

def desenhar_cena(objetos, eye, at, up, fov, aspect_ratio, near, far):
    V, u, v, n = criar_view_matrix_classica(eye, at, up)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    desenhar_camera(ax, eye, at, fov, aspect_ratio, near, far, u, v, n)

    center_distance = (near + far) / 2
    center_world = np.array(eye) - n * center_distance

    transforms = [
        criar_model_matrix(scale=0.5, translation=center_world + [1, 3, 0]),
        criar_model_matrix(scale=0.5, translation=center_world + [2, 1, 0]),
        criar_model_matrix(scale=0.5, translation=center_world + [2, 3, 0]),
        criar_model_matrix(scale=0.5, translation=center_world + [0.5, 1.9, 0]),
        criar_model_matrix(scale=0.5, translation=center_world + [0, 0, 0]),
    ]

    for obj, M_model in zip(objetos, transforms):
        vertices = obj.vertices
        faces = obj.faces
        M = V @ M_model
        vertices_cam = aplicar_transformacao(vertices, M)

        z = vertices_cam[:, 2]
        x = vertices_cam[:, 0]
        y = vertices_cam[:, 1]
        h = 2 * np.tan(np.radians(fov / 2)) * (-z)
        w = h * aspect_ratio
        mask = (np.abs(x) <= w/2) & (np.abs(y) <= h/2) & (-z >= near) & (-z <= far)

        if np.any(mask):
            valid_faces = [face for face in faces if all(mask[i] for i in face)]
            if valid_faces:
                poly3d = Poly3DCollection([vertices_cam[face] for face in valid_faces], facecolors='cyan', edgecolors='k', alpha=0.5)
                ax.add_collection3d(poly3d)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Cena 3D com Base da Câmera Clássica")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

