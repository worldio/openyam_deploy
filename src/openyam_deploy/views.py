"""The views the policy sees: each camera's raw frame reprojected to a pinhole image by its recipe, the config's
``raw`` lens and ``view`` (README.md, Calibrate), and placed by its calibration.

The recipe: maps = cv2.(fisheye.)initUndistortRectifyMap(raw K, raw D, view R.T, view K, view size), then
cv2.remap. The calibration (``calibration.optical_in_parent``) is the view's optical frame (OpenCV axes) in the
camera's parent body, so the view's pose is FK(parent) @ optical_in_parent.
"""
import functools

import numpy as np

from .geometry import CAMERA_FLIP

MAX_RAY_DEG = 85.0       # a fisheye ray further off its lens axis is refused (past 90 deg the model folds it forward),
                         # and so is one past where the lens model's radius stops growing (fisheye_edge)


def camera_models(cam, entry, need_view):
    """A camera's raw lens and view from its config ``entry``, K / D / R as arrays; a SystemExit names what is missing
    or malformed. Without ``need_view`` (a lens capture) only raw.lens is needed."""
    raw = dict(entry.get('raw') or {})
    if raw.get('lens') not in ('fisheye', 'pinhole'):
        raise SystemExit(f'{cam}: the config needs "raw": {{"lens": "fisheye" | "pinhole", ...}}')
    if not need_view:
        return raw, None
    view = dict(entry.get('view') or {})
    missing = ([f'raw.{k}' for k in ('size', 'K', 'D') if k not in raw]
               + [f'view.{k}' for k in ('size', 'K', 'R') if k not in view])
    if missing:
        hint = (' (a lens capture, --intrinsics-only, measures raw.K/D/size)'
                if any(m.startswith('raw.') for m in missing) else '')
        raise SystemExit(f'{cam}: the config lacks {missing}{hint}')
    try:
        raw['K'], raw['D'] = np.array(raw['K'], float).reshape(3, 3), np.array(raw['D'], float).reshape(-1, 1)
        view['K'], view['R'] = np.array(view['K'], float).reshape(3, 3), np.array(view['R'], float).reshape(3, 3)
        raw['size'], view['size'] = [int(v) for v in raw['size']], [int(v) for v in view['size']]
    except (TypeError, ValueError) as error:
        raise SystemExit(f'{cam}: malformed raw / view: {error}') from None
    problems = [name for name, K in (('raw.K', raw['K']), ('view.K', view['K']))
                if not np.allclose(K[2], [0.0, 0.0, 1.0]) or K[0, 0] <= 0 or K[1, 1] <= 0]
    problems += [name for name, size in (('raw.size', raw['size']), ('view.size', view['size']))
                 if len(size) != 2 or min(size) <= 0]
    if raw['lens'] == 'fisheye' and len(raw['D']) != 4:
        problems.append('raw.D (a fisheye lens has 4 coefficients)')
    R = view['R']
    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-5) and np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        problems.append('view.R (not a rotation matrix)')
    if problems:
        raise SystemExit(f'{cam}: malformed {problems}')
    return raw, view


@functools.lru_cache(maxsize=8)
def fisheye_edge(d):
    """The largest distorted radius (normalized image coordinates) a fisheye lens with coefficients ``d`` maps back to
    one ray: its model's theta_d out to MAX_RAY_DEG, cut where it stops growing. A lens fitted with negative high-order
    terms turns over well short of 85 deg (the wrist webcam's at 68), and past that a radius names no single ray."""
    th = np.radians(np.linspace(0.0, MAX_RAY_DEG, 8501))
    thd = th * (1 + d[0] * th ** 2 + d[1] * th ** 4 + d[2] * th ** 6 + d[3] * th ** 8)
    falling = np.flatnonzero(np.diff(thd) <= 0)
    return float(thd[falling[0]] if len(falling) else thd[-1])


def to_view(points, raw, view, bounds=True):
    """(N, 2) raw pixels -> (N, 2) view pixels, and which of them the view sees. Each point goes through the raw lens to
    its ray, into the view's axes (view R: the view's axes in the raw optical frame) and through the view's pinhole K;
    the view sees it if it lies in front of the view and (with ``bounds``) inside its size. A fisheye ray more than
    MAX_RAY_DEG off the lens axis is refused."""
    import cv2
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    K, D = raw['K'], raw['D']
    if raw['lens'] == 'fisheye':
        edge = fisheye_edge(tuple(float(v) for v in D.ravel()))
        seen = np.hypot((pts[:, 0, 0] - K[0, 2]) / K[0, 0], (pts[:, 0, 1] - K[1, 2]) / K[1, 1]) < edge
        rays = cv2.fisheye.undistortPoints(pts, K, D)
    else:
        seen = np.ones(len(pts), bool)
        rays = cv2.undistortPoints(pts, K, D)
    rays = np.c_[rays.reshape(-1, 2), np.ones(len(pts))] @ view['R']         # per row R^T ray: the ray in view axes
    seen &= rays[:, 2] > 1e-9
    uvw = rays @ view['K'].T
    uv = uvw[:, :2] / np.where(seen, uvw[:, 2], 1.0)[:, None]
    w, h = view['size']
    if bounds:
        seen &= (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)
    return uv, seen


def view_maps(raw, view, scale=1):
    """cv2.remap maps from the raw frame to the view, drawn at ``scale`` times the view's size."""
    import cv2
    K = view['K'].copy()
    K[:2] *= scale
    K[:2, 2] = (view['K'][:2, 2] + 0.5) * scale - 0.5                 # pixel centres: the same image, finer
    size = (int(view['size'][0] * scale), int(view['size'][1] * scale))
    rectify = cv2.fisheye.initUndistortRectifyMap if raw['lens'] == 'fisheye' else cv2.initUndistortRectifyMap
    return rectify(raw['K'], raw['D'], view['R'].T, K, size, cv2.CV_32FC1)


class View:
    """A calibrated view, as the server serves it: ``render`` reprojects the camera's raw frame by the recipe, and
    ``c2w`` places it (the view's WODS camera axes -> the kinematics' reference body, 4x4; det -1)."""

    def __init__(self, name, entry):
        self.raw, self.view = camera_models(name, entry, need_view=True)
        calibration = entry.get('calibration') or {}
        if 'optical_in_parent' not in calibration:
            raise SystemExit(f'{name}: not calibrated (no calibration.optical_in_parent); see README.md, Calibrate')
        self.name, self.parent = name, entry['parent_body']
        self.optical_in_parent = np.asarray(calibration['optical_in_parent'], float)
        self.size = tuple(self.view['size'])                  # (w, h)
        self._maps = view_maps(self.raw, self.view)

    def render(self, frame):
        """Raw camera RGB -> the view's RGB (h, w, 3)."""
        import cv2
        h, w = frame.shape[:2]
        if [w, h] != self.raw['size']:
            raise ValueError(f'{self.name}: the camera streams {w}x{h}, but its lens was measured at {self.raw["size"]}')
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def c2w(self, kin):
        T = kin.pose(self.parent) @ self.optical_in_parent
        T[:3, :3] = T[:3, :3] @ CAMERA_FLIP
        return T
