"""
render.py — função show_scene(...)
Agora com:
  • cores individuais também no wireframe
  • parâmetro voxel_size para controle de densidade
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from algebra import apply_transform, camera_R_T, perspective_matrix, project_ndc_to_screen
import math

from meshing import generate_mesh

PALETTE = ["tab:blue", "tab:orange", "tab:green",
           "tab:red",  "tab:purple", "tab:brown"]

def show_individual_solids(
    solids,
    meshes,
    *,
    show_faces=True,
    show_wire=True,
    show_mesh=False,
    palette=PALETTE,
):
    """
    Plota cada sólido em uma janela 3‑D separada, usando somente as
    coordenadas locais.  *meshes* é obrigatório e deve conter uma tupla
    (name, mesh_vertices, mesh_faces) para cada sólido exceto 'line'.
    """
    # ---- dicionário para acesso rápido às malhas ----------------------
    mesh_lookup = {n: (v, f) for n, v, f in meshes}

    for idx, (name, verts, edges) in enumerate(solids, 1):
        color = palette[(idx - 1) % len(palette)]

        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_title(name)

        # ---------- faces preenchidas ----------------------------------
        if name != "line" and show_faces:
            mv, mf = mesh_lookup[name]          # agora é obrigatório existir
            if len(mf):
                tris = [mv[f] for f in mf]
                ax.add_collection3d(
                    Poly3DCollection(tris, facecolors=color,
                                     edgecolors="none", alpha=0.65)
                )

        # ---------- malha (arestas dos triângulos) --------------------
        if name != "line" and show_mesh:
            mv, mf = mesh_lookup[name]
            if len(mf):
                mesh_lines = []
                for a, b, c in mf:
                    mesh_lines += [[mv[a], mv[b]],
                                   [mv[b], mv[c]],
                                   [mv[c], mv[a]]]
                ax.add_collection3d(
                    Line3DCollection(np.array(mesh_lines),
                                     colors=color, lw=0.6)
                )

        # ---------- wireframe ou linha --------------------------------
        if show_wire or name == "line":
            segs = np.array([[verts[i], verts[j]] for i, j in edges])
            ax.add_collection3d(
                Line3DCollection(segs, colors=color, lw=0.8)
            )

        # ---------- enquadramento -------------------------------------
        bb_min, bb_max = verts.min(0), verts.max(0)
        center = (bb_min + bb_max) / 2
        radius = max(bb_max - bb_min) / 2 or 1.0
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect([1,1,1])

        plt.tight_layout()
        plt.show()

def show_scene(
    solids, meshes, transforms,
    show_faces=True,
    show_wire=False,
    show_mesh=False,
    palette=PALETTE,
):
    segments_always, segments_cond = [], []   # ← separa linha dos demais
    facesets = []

    for idx, (name, verts, edges) in enumerate(solids):
        M       = transforms[name]
        col     = palette[idx % len(palette)]
        vw      = apply_transform(verts, M)

        if name == "line":                    # → sempre desenha
            for i, j in edges:
                segments_always.append([[vw[i], vw[j]], col])
            continue                          # não tem faces nem malha

        # wire “condicional”
        if show_wire:
            for i, j in edges:
                segments_cond.append([[vw[i], vw[j]], col])

        # malha / faces
        mv, mf = next((v, f) for n, v, f in meshes if n == name)
        if len(mf):
            mv_w = apply_transform(mv, M)
            tris = [mv_w[k] for k in mf]
            facesets.append((tris, col))

    # -------- plot -----------------------------------------------------
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')

    # faces preenchidas
    if show_faces:
        for tris, col in facesets:
            ax.add_collection3d(
                Poly3DCollection(tris, facecolors=col,
                                 edgecolors='none', alpha=0.6))

    # malha (arestas dos triângulos)
    if show_mesh:
        mesh_lines, mesh_cols = [], []
        for tris, col in facesets:
            for a,b,c in tris:
                mesh_lines += [[a,b],[b,c],[c,a]]
                mesh_cols  += [col,col,col]
        if mesh_lines:
            ax.add_collection3d(
                Line3DCollection(np.array(mesh_lines),
                                 colors=mesh_cols, lw=0.6))

    # wireframe: linha (sempre) + demais (condicionais)
    segs = segments_always + (segments_cond if show_wire else [])
    if segs:
        coords = np.array([p for p,_ in segs])
        cols   = [c for _,c in segs]
        ax.add_collection3d(Line3DCollection(coords, colors=cols, lw=0.8))

    ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_zlim(-10,10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Cena 3-D — faces / wire / malha (linha sempre visível)')
    plt.tight_layout(); plt.show()






def show_camera_scene(
    solids, meshes, world_xforms,
    eye, target,
    fov_y=60, near=1.0, far=10.0,
    show_faces=True, show_wire=False, show_mesh=False,
    palette=PALETTE,
):
    """
    Sistema da CÂMERA:
      • objetos    (faces, wire, malha)
      • frustum    (linhas vermelhas)
      • eye        (vermelho)   e  origem do mundo (amarelo)
    """

    R_cam, T_cam, V = camera_R_T(eye, target)

    segments_always, segments_cond = [], []
    facesets = []

    for idx, (name, verts, edges) in enumerate(solids):
        col = palette[idx % len(palette)]
        vw  = apply_transform(verts, world_xforms[name])
        vc  = apply_transform(vw, V)

        if name == "line":
            for i,j in edges:
                segments_always.append([[vc[i], vc[j]], col])
            continue

        if show_wire:
            for i,j in edges:
                segments_cond.append([[vc[i], vc[j]], col])

        mv, mf = next((v,f) for n,v,f in meshes if n==name)
        if len(mf):
            mv_c = apply_transform(apply_transform(mv, world_xforms[name]), V)
            tris = [mv_c[k] for k in mf]
            facesets.append((tris, col))

    # ---------- plotagem ----------------------------------------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    fig = plt.figure(figsize=(6,6), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')

    # faces preenchidas
    if show_faces:
        for tris, col in facesets:
            ax.add_collection3d(
                Poly3DCollection(tris, facecolors=col,
                                 edgecolors='none', alpha=0.6)
            )

    # malha (arestas dos triângulos)
    if show_mesh:
        mesh_lines, mesh_cols = [], []
        for tris, col in facesets:
            for a,b,c in tris:
                mesh_lines += [[a,b], [b,c], [c,a]]
                mesh_cols  += [col,col,col]
        if mesh_lines:
            ax.add_collection3d(
                Line3DCollection(np.array(mesh_lines),
                                 colors=mesh_cols, lw=0.6)
            )

    segs = segments_always + (segments_cond if show_wire else [])
    if segs:
        coords = np.array([p for p,_ in segs])
        cols   = [c for _,c in segs]
        ax.add_collection3d(Line3DCollection(coords, colors=cols, lw=0.8))

    # -------- frustum e pontos especiais ------------------------------
    h_n = 2*np.tan(np.deg2rad(fov_y/2))*near
    h_f = 2*np.tan(np.deg2rad(fov_y/2))*far
    w_n = h_n; w_f = h_f
    cn  = np.array([0,0,-near]); cf = np.array([0,0,-far])
    N = cn + np.array([[-w_n/2,-h_n/2,0],[ w_n/2,-h_n/2,0],
                       [ w_n/2, h_n/2,0],[-w_n/2, h_n/2,0]])
    F = cf + np.array([[-w_f/2,-h_f/2,0],[ w_f/2,-h_f/2,0],
                       [ w_f/2, h_f/2,0],[-w_f/2, h_f/2,0]])
    fr = [[np.zeros(3),p] for p in N] + \
         [[N[i],N[(i+1)%4]] for i in range(4)] + \
         [[F[i],F[(i+1)%4]] for i in range(4)] + \
         [[N[i],F[i]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(np.array(fr), colors='r', lw=1.0))

    world_origin_c = apply_transform(np.array([[0,0,0]]), V)[0]
    ax.scatter(0,0,0,               color='red',    s=50, label='Camera')
    ax.scatter(*world_origin_c,     color='yellow', s=40, label='World 0')

    lim = far
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, 0)
    ax.set_xlabel('Xc'); ax.set_ylabel('Yc'); ax.set_zlabel('Zc')
    ax.set_title('Espaço da CÂMERA')
    ax.legend()
    plt.tight_layout(); plt.show()
    
    
    
    
    
    
    
    
def show_projection_demonstration_final(
    solids, meshes, world_xforms,
    eye, target,
    fov_y=60, near=1.0, far=10.0,
    show_faces=True,
    show_mesh=False,
    palette=PALETTE,
):
    R_cam, T_cam, V = camera_R_T(eye, target)
    d_proj = near + 0.25 * (far - near)        # plano “papel”

    # ---------- coleta em espaço‑câmera --------------------------------
    camera_objs = []
    line_data   = None
    for idx, (name, verts, edges) in enumerate(solids):
        color = palette[idx % len(palette)]
        if name == "line":
            v_line = apply_transform(apply_transform(verts, world_xforms[name]), V)
            line_data = (v_line, edges, color)
            continue

        mv, mf = next((v,f) for n,v,f in meshes if n==name)
        if len(mf):
            mv_c = apply_transform(apply_transform(mv, world_xforms[name]), V)
            camera_objs.append({'verts':mv_c, 'faces':mf, 'color':color})

    # ---------- plot ---------------------------------------------------
    fig = plt.figure(figsize=(9,7), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-75)

    # 1) objetos 3‑D originais
    if show_faces:
        for obj in camera_objs:
            tris = [obj['verts'][f] for f in obj['faces']]
            ax.add_collection3d(
                Poly3DCollection(tris, facecolors=obj['color'],
                                 edgecolors='none', alpha=0.6, shade=False)
            )
    if show_mesh:
        for obj in camera_objs:
            lines = []
            for a,b,c in [obj['verts'][f] for f in obj['faces']]:
                lines += [[a,b],[b,c],[c,a]]
            ax.add_collection3d(
                Line3DCollection(np.array(lines),
                                 colors=obj['color'], lw=0.6)
            )

    # linha (objeto sempre wire)
    if line_data:
        v_line, edges, col = line_data
        for i,j in edges:
            ax.add_collection3d(
                Line3DCollection([[v_line[i], v_line[j]]],
                                 colors=col, lw=1.2)
            )

    # 2) retângulo do plano de projeção
    h_p = 2 * d_proj * np.tan(np.deg2rad(fov_y/2))
    w_p = h_p
    plane_center = np.array([0,0,-d_proj])
    plane = plane_center + np.array(
        [[-w_p/2,-h_p/2,0],[ w_p/2,-h_p/2,0],
         [ w_p/2, h_p/2,0],[-w_p/2, h_p/2,0]]
    )
    ax.add_collection3d(
        Poly3DCollection([plane], facecolors='white',
                         edgecolors='k', alpha=0.5, shade=False)
    )

    # 3) projeção de cada objeto sobre o plano
    for obj in camera_objs:
        v   = obj['verts']
        col = obj['color']
        front = v[:,2] < -1e-5
        if not front.any(): continue

        proj = np.copy(v)
        zc   = proj[front,2]
        proj[front,0] = proj[front,0] * d_proj / -zc
        proj[front,1] = proj[front,1] * d_proj / -zc
        proj[front,2] = -d_proj

        tris_proj = []
        for f in obj['faces']:
            if all(front[i] for i in f):
                tris_proj.append([proj[i] for i in f])

        if tris_proj:
            ax.add_collection3d(
                Poly3DCollection(tris_proj, facecolors=col,
                                 edgecolors='none', alpha=1.0, shade=False)
            )

    # projeção da linha
    if line_data:
        v_line, edges, col = line_data
        proj_line = []
        for p in v_line:
            if p[2] < -1e-5:
                proj_line.append([p[0]*d_proj/-p[2],
                                  p[1]*d_proj/-p[2],
                                  -d_proj])
            else:
                proj_line.append(p)  # invisível; não desenha
        for i,j in edges:
            ax.add_collection3d(
                Line3DCollection([[proj_line[i], proj_line[j]]],
                                 colors=col, lw=1.2)
            )

    # 4) frustum + pontos
    world_origin_c = apply_transform(np.array([[0,0,0]]), V)[0]
    ax.scatter(0,0,0,           color='red',    s=50, label='Camera')
    ax.scatter(*world_origin_c, color='yellow', s=40, label='World 0')

    h_n, h_f = 2*np.tan(np.deg2rad(fov_y/2))*near, 2*np.tan(np.deg2rad(fov_y/2))*far
    w_n, w_f = h_n, h_f
    cn, cf = np.array([0,0,-near]), np.array([0,0,-far])
    N = cn + np.array([[-w_n/2,-h_n/2,0],[ w_n/2,-h_n/2,0],
                       [ w_n/2, h_n/2,0],[-w_n/2, h_n/2,0]])
    F = cf + np.array([[-w_f/2,-h_f/2,0],[ w_f/2,-h_f/2,0],
                       [ w_f/2, h_f/2,0],[-w_f/2, h_f/2,0]])
    fr = [[N[i],N[(i+1)%4]] for i in range(4)] + \
         [[F[i],F[(i+1)%4]] for i in range(4)] + \
         [[N[i],F[i]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(np.array(fr), colors='r', lw=1.0))

    # limites
    lim = far*1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, 0)
    ax.set_xlabel('Xc'); ax.set_ylabel('Yc'); ax.set_zlabel('Zc')
    ax.set_title('Projeção no plano (cores preservadas + linha)')
    ax.legend(); plt.tight_layout(); plt.show()
    
    
    
    
    
    
    # ----------------------------------------------------------------------
# Projeção 2‑D mostrando só a pirâmide menor + plano da janela
# ----------------------------------------------------------------------
def show_projection_window_only(
    solids, meshes, world_xforms,
    eye, target,
    fov_y=60, near=1.0, far=10.0,
    palette=PALETTE,
):
    R_cam, T_cam, V = camera_R_T(eye, target)
    d_proj = near + 0.25 * (far - near)          # mesmo plano-papel

    # ---------- coleta em espaço‑câmera --------------------------------
    objs, line_data = [], None
    for idx, (name, verts, edges) in enumerate(solids):
        col = palette[idx % len(palette)]
        if name == "line":
            v_line = apply_transform(apply_transform(verts, world_xforms[name]), V)
            line_data = (v_line, edges, col)
            continue
        mv, mf = next((v,f) for n,v,f in meshes if n==name)
        if len(mf):
            mv_c = apply_transform(apply_transform(mv, world_xforms[name]), V)
            objs.append({'v':mv_c, 'f':mf, 'c':col})

    # ---------- plot básico -------------------------------------------
    fig = plt.figure(figsize=(8,6), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-75)

    # retângulo da janela (plano de projeção)
    h_p = 2*d_proj*np.tan(np.deg2rad(fov_y/2)); w_p = h_p
    plane_c = np.array([0,0,-d_proj])
    rect = plane_c + np.array(
        [[-w_p/2,-h_p/2,0],[ w_p/2,-h_p/2,0],
         [ w_p/2, h_p/2,0],[-w_p/2, h_p/2,0]]
    )
    ax.add_collection3d(Poly3DCollection([rect], facecolors='white',
                                         edgecolors='k', alpha=.4, shade=False))

    # arestas da pirâmide menor
    edges_small = [[np.zeros(3), rect[i]] for i in range(4)] + \
                  [[rect[i], rect[(i+1)%4]] for i in range(4)]
    ax.add_collection3d(Line3DCollection(edges_small, colors='k', lw=1.2))

    # ---------- projeção dos objetos ----------------------------------
    for obj in objs:
        v  = obj['v']; col = obj['c']
        front = v[:,2] < -1e-5
        if not front.any(): continue
        proj = np.copy(v)
        zc   = proj[front,2]
        proj[front,0] = proj[front,0] * d_proj / -zc
        proj[front,1] = proj[front,1] * d_proj / -zc
        proj[front,2] = -d_proj
        tris = []
        for f in obj['f']:
            if all(front[i] for i in f):
                tris.append([proj[i] for i in f])
        if tris:
            ax.add_collection3d(
                Poly3DCollection(tris, facecolors=col,
                                 edgecolors='none', alpha=1.0, shade=False)
            )

    # ---------- projeção da linha -------------------------------------
    if line_data:
        v_line, edges, col = line_data
        proj_line = []
        for p in v_line:
            if p[2] < -1e-5:
                proj_line.append([p[0]*d_proj/-p[2],
                                  p[1]*d_proj/-p[2],
                                  -d_proj])
            else:
                proj_line.append(p)          # invisível
        for i,j in edges:
            if proj_line[i][2] == -d_proj and proj_line[j][2] == -d_proj:
                ax.add_collection3d(
                    Line3DCollection([[proj_line[i], proj_line[j]]],
                                     colors=col, lw=1.2)
                )

    # ---------- pontos especiais --------------------------------------
    ax.scatter(0,0,0, color='red', s=50, label='Camera')
    ax.set_xlim(-w_p, w_p); ax.set_ylim(-w_p, w_p); ax.set_zlim(-2*d_proj, 0)
    ax.set_xlabel('Xc'); ax.set_ylabel('Yc'); ax.set_zlabel('Zc')
    ax.set_title('Janela de projeção + pirâmide menor (2‑D projetado)')
    ax.legend(); plt.tight_layout(); plt.show()


    
    
    
    
    
# ----------------------------------------------------------------------
# Rasterização rápida usando Pillow (polígonos + linhas)
# ----------------------------------------------------------------------
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def _rgb(color):            # cor matplotlib -> (R,G,B)
    r,g,b,_ = mcolors.to_rgba(color)
    return int(r*255), int(g*255), int(b*255)

def rasterize_projection_scene(
    solids, meshes, world_xforms,
    eye, target,
    fov_y=60, near=1.0, far=10.0,
    resolutions=(256, 512, 1024),
    palette=PALETTE,
):
    _, _, V = camera_R_T(eye, target)
    P       = perspective_matrix(fov_y, 1.0, near, far)

    # ---------------- pré-projeção de TODOS os vértices ----------------
    proj_cache = {}   # (name) -> dict{ 'verts': scr_xy , 'mask': in_NDC }
    for name, verts, _ in solids:
        M     = P @ V @ world_xforms[name]
        vh    = np.hstack([verts, np.ones((len(verts),1))])
        clip  = (M @ vh.T).T
        w     = clip[:,3]
        ndc   = clip[:,:3] / w[:,None]
        mask  = (w>0) & (np.abs(ndc)<=1).all(axis=1)
        proj_cache[name] = {'ndc':ndc, 'mask':mask}

    # ------------------------- por resolução ---------------------------
    for res in resolutions:
        img   = Image.new("RGB", (res, res), (255,255,255))
        draw  = ImageDraw.Draw(img)

        for idx, (name, verts, edges) in enumerate(solids):
            col  = palette[idx % len(palette)]
            rgb  = _rgb(col)
            ndc  = proj_cache[name]['ndc']
            mask = proj_cache[name]['mask']
            if not mask.any():     # nada visível nesta primitiva
                continue

            scr = project_ndc_to_screen(ndc[mask,:2], res, res)
            idx_scr = {old:new for new,old in enumerate(np.where(mask)[0])}

            # ------ linha (sempre) ------------------------------------
            if name == "line":
                for i,j in edges:
                    if mask[i] and mask[j]:
                        p,q = scr[idx_scr[i]], scr[idx_scr[j]]
                        draw.line((*p, *q), fill=rgb) #brensenham
                continue

            # ------ faces da malha ------------------------------------
            mv, mf = next((v,f) for n,v,f in meshes if n==name)
            mvh    = np.hstack([mv, np.ones((len(mv),1))])
            clip_m = (P @ V @ world_xforms[name] @ mvh.T).T
            w_m    = clip_m[:,3]
            ndc_m  = clip_m[:,:3] / w_m[:,None]
            ok     = (w_m>0) & (np.abs(ndc_m)<=1).all(axis=1)
            if not ok.any(): continue
            scr_m  = project_ndc_to_screen(ndc_m[ok,:2], res, res)
            idxm   = {old:new for new,old in enumerate(np.where(ok)[0])}

            for a,b,c in mf:
                if ok[a] and ok[b] and ok[c]:
                    pa,pb,pc = scr_m[idxm[a]], scr_m[idxm[b]], scr_m[idxm[c]]
                    draw.polygon([tuple(pa),tuple(pb),tuple(pc)], fill=rgb) #scan line

        fname = f"raster_{res}.png"
        img.save(fname);   print(f"[OK] {fname} salvo.")

        # ---------- mostra na tela ------------------------------------
        plt.figure(figsize=(res/100, res/100), dpi=100)
        plt.imshow(img); plt.axis('off')
        plt.title(f"Raster {res}×{res}")
        plt.tight_layout(); plt.show()
