"""The node graph drawn over the views, as omniact-molmospaces draws it: ``project_cam_frame``, ``get_font``,
``render_view`` and ``pose_points`` are verbatim from its scripts/embodiments/visualize_node_graph.py, and
``strip`` lays one observation out as that script lays out a frame: [skeleton | view_1 | view_2 | ...], the skeleton
panel the graph alone in the view that sees most of it, each node's pose (the TCP's) as x / y / z axes.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .embodiment import JOINT_PRISMATIC, JOINT_REVOLUTE

CV_TO_WODS = np.diag([1.0, -1.0, 1.0])   # wods<->OpenCV camera flip (for camera-frame geom): only y, down<->up

EDGE, AX, REV, PRIS, TERM = (210, 210, 215), (110, 150, 255), (255, 235, 40), (60, 190, 255), (40, 255, 120)
POSE_AXES, POSE_LEN = ((255, 80, 80), (80, 255, 80), (30, 60, 255)), 0.05   # node pose x / y / z tips, metres


def project_cam_frame(pc, K, image_size):
    """(N, P, 3) wods camera-frame points + intrinsics K(3,3) -> (N, P, 3) normalized
    (x_norm, y_norm, depth): flip wods->OpenCV (CV_TO_WODS), pinhole-project, normalize x,y by image
    size (off-frame kept raw). depth = forward camera-z (>0 in front)."""
    pc_cv = np.asarray(pc) @ CV_TO_WODS
    height, width = image_size
    z = pc_cv[..., 2]
    zs = np.where(np.abs(z) > 1e-6, z, 1e-6)
    xn = (K[0, 0] * pc_cv[..., 0] / zs + K[0, 2]) / width
    yn = (K[1, 1] * pc_cv[..., 1] / zs + K[1, 2]) / height
    return np.stack([xn, yn, z], axis=-1)
_FONT_CACHE = {}


def get_font(size):
    if size not in _FONT_CACHE:
        import matplotlib.font_manager as fm
        f = None
        for cand in (fm.findfont("DejaVu Sans"), "DejaVuSans.ttf"):
            try:
                f = ImageFont.truetype(cand, size); break
            except Exception:
                pass
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


def render_view(frame, proj, edges, types, names, W, H, scale=4, margin=0.7, limb=False, title=None, poses=None):
    """proj: per node (center_norm(x,y,depth), axistip_norm or None). Returns HxW3 array.
    ``poses`` (optional): per node its projected pose points (origin, x tip, y tip, z tip).
    margin is (horizontal, vertical) fractions, or a scalar — horizontal kept small so panels sit
    close together, vertical larger to hold off-frame joints. (Verbatim from _graphviz.py.)
    ``title`` (optional) is drawn small in the top margin."""
    mx, my = margin if isinstance(margin, (tuple, list)) else (margin, margin)
    cw, ch = int((1 + 2 * mx) * W * scale), int((1 + 2 * my) * H * scale)
    canvas = Image.new("RGB", (cw, ch), (26, 26, 30))
    ox, oy = int(mx * W * scale), int(my * H * scale)
    if frame is not None:
        canvas.paste(Image.fromarray(frame).resize((W * scale, H * scale), Image.NEAREST), (ox, oy))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((ox + 2, 1), title, fill=(235, 235, 240), font=get_font(max(9, int(7 * scale))))
    d.rectangle([ox, oy, ox + W * scale, oy + H * scale], outline=(125, 125, 130), width=max(1, scale))
    font = get_font(int(8.5 * scale))

    def c2(p):
        return ((mx + p[0]) * W * scale, (my + p[1]) * H * scale)

    def vis(p):
        return p is not None and p[2] > 0 and -mx <= p[0] <= 1 + mx and -my <= p[1] <= 1 + my

    def clip(pa, pb):                                         # Liang-Barsky clip of segment to the canvas
        x0, y0 = c2(pa); x1, y1 = c2(pb); dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0), (dx, cw - x0), (-dy, y0), (dy, ch - y0)):
            if p == 0:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        return None
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        return None
                    t1 = min(t1, r)
        return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)

    lw = max(2, int(round(1.5 * scale)))
    for a, b in edges:                                        # edges drawn even toward off-frame nodes (clipped)
        pa, pb = proj[a][0], proj[b][0]
        if pa[2] > 0 and pb[2] > 0 and (seg := clip(pa, pb)) is not None:
            if limb:                                          # stretched "metal limb" texture
                d.line(seg, fill=(95, 97, 105), width=lw * 5)
                d.line(seg, fill=(175, 178, 188), width=lw * 3)
                d.line(seg, fill=(225, 228, 236), width=lw)
            else:
                d.line(seg, fill=EDGE, width=lw)
    for n in range(len(proj)):
        c, at = proj[n]
        if vis(c) and at is not None and at[2] > 0 and (seg := clip(c, at)) is not None:
            d.line(seg, fill=AX, width=lw)
    for pts in poses or ():
        for tip, col in zip(pts[1:], POSE_AXES):
            if vis(pts[0]) and tip[2] > 0 and (seg := clip(pts[0], tip)) is not None:
                d.line(seg, fill=col, width=lw)
    for n in range(len(proj)):
        c = proj[n][0]
        if not vis(c):
            continue
        u = c2(c); term = types[n] == "term"; col = TERM if term else (PRIS if types[n] == "pris" else REV)
        r = scale * (1.5 if term else 1.35)
        d.ellipse([u[0] - r, u[1] - r, u[0] + r, u[1] + r], fill=col)
        d.text((u[0] + r + 2, u[1] - r - 2 - (n % 2) * int(2.6 * scale)), f"{n}:{names[n]}", fill=col, font=font)
    return np.asarray(canvas)


def pose_points(P, C=None):
    """(T, N, 3, 4) SE(3) poses -> (T, N, 4, 3) camera-frame points (origin, x tip, y tip, z tip); a zero pose
    gives NaNs, which never draw. ``C`` (T, 3, 4) c2w takes a WORLD track into this view's camera frame."""
    M, o = P[..., :3, :3], P[..., :3, 3:]
    pts = np.concatenate([o, o + POSE_LEN * M], axis=-1).swapaxes(-1, -2)
    pts = np.where(np.any(M != 0, axis=(-1, -2))[..., None, None], pts, np.nan)
    return pts if C is None else (pts - C[:, None, None, :3, 3]) @ C[:, None, :3, :3]


def node_types(joint_type):
    return ["pris" if t == JOINT_PRISMATIC else "rev" if t == JOINT_REVOLUTE else "term" for t in joint_type]


def projected(geometry, K, size, axis_mask):
    """One view's (N, 2, 3) camera-frame node geometry -> render_view's ``proj``."""
    g = project_cam_frame(geometry, K, size)
    return [(tuple(map(float, g[n, 0])), tuple(map(float, g[n, 1])) if axis_mask[n] else None)
            for n in range(len(g))]


def strip(obs, intrinsics, scale=2, margin=(0.1, 0.2), titles=None):
    """[skeleton | view_1 | view_2 | ...] for one observation (the server's: ``views``, ``node_geometry``,
    ``node_pose``, the graph, ``camera_names``), ``intrinsics`` a view name -> its K. Returns an RGB array."""
    views, geometry, poses = obs['views'], np.asarray(obs['node_geometry']), np.asarray(obs['node_pose'])
    names, edges = list(obs['node_names']), [tuple(int(v) for v in e) for e in np.asarray(obs['edges'])]
    axis_mask, types = np.asarray(obs['axis_mask'], bool), node_types(np.asarray(obs['joint_type']))
    H, W = views.shape[1:3]
    panels, seen = [], []
    for v, cam in enumerate(obs['camera_names']):
        proj = projected(geometry[:, :, v], intrinsics[cam], (H, W), axis_mask)
        pts = project_cam_frame(pose_points(poses[None, :, v])[0], intrinsics[cam], (H, W))
        panel_poses = [tuple(map(tuple, p)) for p in pts]
        centres = np.array([c for c, _ in proj])
        seen.append(float(((centres[:, 2] > 0) & (centres[:, 0] >= 0) & (centres[:, 0] <= 1)
                           & (centres[:, 1] >= 0) & (centres[:, 1] <= 1)).mean()))
        title = cam if titles is None else titles[v]
        panels.append((views[v], proj, panel_poses, title))
    best = int(np.argmax(seen))
    row = [render_view(None, panels[best][1], edges, types, names, W, H, scale, margin, title='skeleton',
                       poses=panels[best][2])]
    row += [render_view(frame, proj, edges, types, names, W, H, scale, margin, title=title, poses=panel_poses)
            for frame, proj, panel_poses, title in panels]
    return np.concatenate(row, axis=1)
