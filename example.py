"""example.py – demo pipeline"""
import numpy as np
from geometry import (
    create_line, create_cylinder, create_pipe_straight,
    create_box, create_pipe_curved,
)
from meshing import generate_mesh
from algebra import translate, rot_x, rot_y, rot_z, scale, apply_transform
import render


def main(show_faces=True, show_wire=True, show_mesh=True):
    # solids ------------------------------------------------------------
    solids = [
        ("cylinder",      *create_cylinder(1.0, 2.0)),
        ("pipe_straight", *create_pipe_straight(0.2, 0.5, 4.0, radial_segments=32)),
        ("box",           *create_box(6.0, 2.0, 2.0)),
        ("pipe_curved",
         *create_pipe_curved(0.2, 0.5, (0, 0, 0), (4, 4, 0),
                             (4, 0, 0), (0, 4, 0), radial_segments=16)),
        ("line", *create_line(4.0)),
    ]

    # meshes ------------------------------------------------------------
    meshes = []
    for name, verts, edges in solids:
        if name == "line":
            continue
        if name == "pipe_straight":
            mesh_v, mesh_f = generate_mesh(name, verts, edges, radial_segments=32)
        elif name == "pipe_curved":
            mesh_v, mesh_f = generate_mesh(name, verts, edges, radial_segments=16)
        else:
            mesh_v, mesh_f = generate_mesh(name, verts, edges)
        meshes.append((name, mesh_v, mesh_f))

    # transforms --------------------------------------------------------
    deg = np.deg2rad
    transforms = {
        "cylinder":      translate(-4, 0, 0) @ scale(.8, .8, .8),
        "pipe_straight": translate(0, -4, 0) @ rot_y(deg(90)) @ scale(.8, .8, .8),
        "box":           translate(4, 0, 0) @ rot_z(deg(45)) @ scale(.7, .7, .7),
        "pipe_curved":   translate(0, 3, -4) @ rot_x(deg(-30)) @ scale(.5, .5, .5),
        "line":          translate(0, 0, 4) @ rot_z(deg(30)) @ scale(.8, .8, .8),
    }

    # camera setup ------------------------------------------------------
    all_world = [apply_transform(v, transforms[n]) for n, v, _ in solids]
    all_world = np.vstack(all_world)
    bb_min, bb_max = all_world.min(0), all_world.max(0)
    target = (bb_min + bb_max) / 2.0
    eye = bb_max + np.array([2, 2, 2])
    dist = np.linalg.norm(eye - target)
    near, far = max(0.1, dist * 0.05), dist + 5

    # views -------------------------------------------------------------
    render.show_individual_solids(solids, meshes,
                                  show_faces=True, show_wire=False, show_mesh=False)

    render.show_scene(solids, meshes, transforms,
                      show_faces=show_faces, show_wire=show_wire, show_mesh=show_mesh)

    render.show_camera_scene(solids, meshes, transforms,
                             eye=eye, target=target, fov_y=60,
                             near=near, far=far,
                             show_faces=True, show_wire=False, show_mesh=False)

    render.show_projection_demonstration_final(solids, meshes, transforms,
                                               eye=eye, target=target,
                                               fov_y=60, near=near, far=far,
                                               show_faces=True, show_mesh=True)

    render.show_projection_window_only(solids, meshes, transforms,
                                       eye=eye, target=target,
                                       near=near, far=far, fov_y=60)

    render.rasterize_projection_scene(solids, meshes, transforms,
                                      eye=eye, target=target,
                                      fov_y=60, near=near, far=far)


if __name__ == "__main__":
    main(show_faces=True, show_wire=False, show_mesh=True)
