"""
example.py  –  pipeline completo
"""

import numpy as np
from geometry import (
    create_line, create_cylinder, create_pipe_straight,
    create_box,  create_pipe_curved,
)
from meshing import generate_mesh
from algebra  import translate, rot_x, rot_y, rot_z, scale, apply_transform
import render

def main(show_faces=True, show_wire=True, show_mesh=True):

    # ---------- 1. Sólidos locais --------------------------------------
    solids = [
        ("cylinder",      *create_cylinder(1.0, 2.0)),
        ("pipe_straight", *create_pipe_straight(0.5, 4.0)),
        ("box",           *create_box(6.0, 2.0, 2.0)),
    ]
    solids.append(("pipe_curved",
                   *create_pipe_curved(0.5, (0,0,0), (4,4,0), (4,0,0), (0,4,0))))
    solids.append(("line", *create_line(4.0)))          # wireframe only

    # ---------- 2. Mesh --------------------------------------
    meshes = []
    for name, verts, edges in solids:
        if name == "line":
            continue
        mesh_v, mesh_f = generate_mesh(name, verts, edges)
        meshes.append((name, mesh_v, mesh_f))

    # ---------- 3. Transformações (posição final no mundo) ------------
    deg = np.deg2rad
    transforms = {
        "cylinder"     : translate(-4,  0, 0) @ scale(.8,.8,.8),
        "pipe_straight": translate( 0, -4, 0) @ rot_y(deg(90)) @ scale(.8,.8,.8),
        "box"          : translate( 4,  0, 0) @ rot_z(deg(45)) @ scale(.7,.7,.7),
        "pipe_curved"  : translate( 0,  3, -4) @ rot_x(deg(-30)) @ scale(.5,.5,.5),
        "line"         : translate( 0,  0, 4) @ rot_z(deg(30)) @ scale(.8,.8,.8),
    }

    # ---------- 4. Bounding-box global → eye & target ------------------
    all_world = []
    for name, verts, _ in solids:
        vw = apply_transform(verts, transforms[name])  # world vertices
        all_world.append(vw)
    all_world = np.vstack(all_world)

    bb_min, bb_max = all_world.min(0), all_world.max(0)
    target = (bb_min + bb_max) / 2.0                  # centro dos objetos
    eye    = bb_max + np.array([2, 2, 2])             # canto “superior”

    dist = np.linalg.norm(eye - target)
    near = max(0.1, dist * 0.05)         # 5 % da distância
    far  = dist + 5                      # folga extra




    # ---------- 5. Visualizações --------------------------------------
    render.show_individual_solids(
        solids, meshes,
        show_faces=True,
        show_wire=True,
        show_mesh=False   # ative se quiser ver a malha
    )
    
    render.show_scene(solids, meshes, transforms,
                      show_faces=show_faces, show_wire=show_wire, show_mesh=show_mesh)


    render.show_camera_scene(
        solids, meshes, transforms,
        eye=eye, target=target,
        fov_y=60, near=near, far=far,    # ← usa valores auto
        show_faces=True, show_wire=False, show_mesh=False,
    )
    
    render.show_projection_demonstration_final(
        solids, meshes, transforms,
        eye=eye, target=target,
        fov_y=60, near=near, far=far,
        show_faces=True,   # Desenha os objetos 3D com faces preenchidas
        show_mesh=True,   # Desenha a malha sobre as faces para mais detalhes
    )
    
    render.show_projection_window_only(
        solids, meshes, transforms,
        eye=eye, target=target,
        near=near, far=far, fov_y=60
    )
    
    
    render.rasterize_projection_scene(
        solids, meshes, transforms,
        eye=eye, target=target,
        fov_y=60, near=near, far=far
    )

if __name__ == "__main__":
    main(show_faces=True, show_wire=False, show_mesh=True)

