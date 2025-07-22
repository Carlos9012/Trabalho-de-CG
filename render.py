"""
render.py — função show_scene(...)
Agora com:
  • cores individuais também no wireframe
  • parâmetro voxel_size para controle de densidade
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from algebra import apply_transform, camera_R_T, perspective_matrix, project_ndc_to_screen

PALETTE = ["tab:blue", "tab:orange", "tab:green",
           "tab:red",  "tab:purple", "tab:brown"]

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
    
    
    
    
    
    
    
    
def show_projection_scene(
    solids, meshes, world_xforms,
    eye, target,
    fov_y=60, near=1.0, far=10.0,
    img_size=(800, 800),
    show_faces=False, show_wire=True, show_mesh=False,
    palette=None,
):
    if palette is None:
        palette = ["tab:blue", "tab:orange", "tab:green",
                   "tab:red",  "tab:purple", "tab:brown"]

    _, _, V = camera_R_T(eye, target)
    aspect  = img_size[0] / img_size[1]
    P       = perspective_matrix(fov_y, aspect, near, far)

    fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
    ax.set_xlim(0, img_size[0]); ax.set_ylim(img_size[1], 0)
    ax.set_aspect('equal'); ax.axis('off')

    for idx, (name, verts, edges) in enumerate(solids):
        col = palette[idx % len(palette)]
        M   = P @ V @ world_xforms[name]

        # -------- vértices em clip space -------------------------------
        v_h   = np.hstack([verts, np.ones((len(verts), 1))])      # (N,4)
        clip  = (M @ v_h.T).T
        w     = clip[:, 3]

        # mantêm só pontos dentro do cubo NDC
        ndc   = clip[:, :3] / w[:, None]
        in_ndc = (w > 0) & (np.abs(ndc[:,0]) <= 1) & \
                 (np.abs(ndc[:,1]) <= 1) & (np.abs(ndc[:,2]) <= 1)

        if not in_ndc.any():
            continue

        screen = np.empty((len(verts), 2))
        screen[in_ndc] = project_ndc_to_screen(ndc[in_ndc, :2], *img_size)
        idx_scr = {old: new for new, old in enumerate(np.where(in_ndc)[0])}

        # -------- LINHA (sempre mostrada) ------------------------------
        if name == "line":
            for i, j in edges:
                if in_ndc[i] and in_ndc[j]:
                    p, q = idx_scr[i], idx_scr[j]
                    ax.plot([screen[p,0], screen[q,0]],
                            [screen[p,1], screen[q,1]],
                            color=col, lw=1)
            continue

        # -------- wireframe lógico ------------------------------------
        if show_wire:
            for i, j in edges:
                if in_ndc[i] and in_ndc[j]:
                    p, q = idx_scr[i], idx_scr[j]
                    ax.plot([screen[p,0], screen[q,0]],
                            [screen[p,1], screen[q,1]],
                            color=col, lw=1)

        # -------- faces / malha ---------------------------------------
        if name != "line" and (show_faces or show_mesh):
            mv, mf = next((v, f) for n, v, f in meshes if n == name)
            mv_h   = np.hstack([mv, np.ones((len(mv),1))])
            clip_m = (M @ mv_h.T).T
            w_m    = clip_m[:,3]

            ndc_m  = clip_m[:, :3] / w_m[:, None]
            in_m   = (w_m > 0) & (np.abs(ndc_m[:,0]) <= 1) & \
                     (np.abs(ndc_m[:,1]) <= 1) & (np.abs(ndc_m[:,2]) <= 1)
            if not in_m.any():
                continue

            scr_m = project_ndc_to_screen(ndc_m[in_m,:2], *img_size)
            idx_scr_m = {old:new for new,old in enumerate(np.where(in_m)[0])}

            for a, b, c in mf:
                if not (in_m[a] and in_m[b] and in_m[c]):
                    continue
                pa, pb, pc = idx_scr_m[a], idx_scr_m[b], idx_scr_m[c]

                if show_faces:
                    ax.fill([scr_m[pa,0], scr_m[pb,0], scr_m[pc,0]],
                            [scr_m[pa,1], scr_m[pb,1], scr_m[pc,1]],
                            facecolor=col, edgecolor='none', alpha=0.3)

                if show_mesh:
                    ax.plot([scr_m[pa,0], scr_m[pb,0]],
                            [scr_m[pa,1], scr_m[pb,1]],
                            color=col, lw=0.6)
                    ax.plot([scr_m[pb,0], scr_m[pc,0]],
                            [scr_m[pb,1], scr_m[pc,1]],
                            color=col, lw=0.6)
                    ax.plot([scr_m[pc,0], scr_m[pa,0]],
                            [scr_m[pc,1], scr_m[pa,1]],
                            color=col, lw=0.6)

    ax.set_title('Projeção em perspectiva 2-D (clipped em NDC)')
    plt.tight_layout(); plt.show()
    
    
    
    
    
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
