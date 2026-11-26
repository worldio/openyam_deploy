"""Rigid transforms, and the axis conventions of the policy's wire (omniact-molmospaces'):

* cameras: a calibration stores the OpenCV optical frame (x right, y down, z forward); the policy's camera frame is
  WODS's, y up (``CAMERA_FLIP``), so its c2w has det -1;
* TCP: the ``gripper/tcp_frame`` site is GL-style (x toward the right pad, y up, z back into the wrist); the policy
  sees its WODS axes, z forward into the fingertips (``GL_TO_WODS``, an involution: it converts both ways).
"""
import numpy as np

CAMERA_FLIP = np.diag([1.0, -1.0, 1.0])
GL_TO_WODS = np.diag([1.0, 1.0, -1.0])


def inv_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4); out[:3, :3] = R.T; out[:3, 3] = -R.T @ t
    return out


def rt_to_se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).reshape(3)
    return T
