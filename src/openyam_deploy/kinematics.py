"""MuJoCo forward kinematics of the robot's live URDF (/robot_description).

``build_model`` / ``fk_poses`` are the plain FK the calibration uses. ``Kinematics`` is the server's: it reports every
frame in the config's ``reference_body`` (the arm base: the "world" of the policy's wire, ``c2w`` and
``node_action_pose``), follows the URDF's mimic joints, and carries the ``gripper/tcp_frame`` site the IK solves for.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import inv_se3

TCP_SITE = 'gripper/tcp_frame'


def _bare_urdf(urdf_text):
    """The URDF's XML with visual/collision elements stripped (they carry package:// meshes mujoco can't resolve;
    FK needs only the joint tree), visuals discarded and fixed links kept as named bodies."""
    root = ET.fromstring(urdf_text)
    for link in root.findall('link'):
        for kind in ('visual', 'collision'):
            for item in list(link.findall(kind)):
                link.remove(item)
    ext = root.find('mujoco'); ext = ET.SubElement(root, 'mujoco') if ext is None else ext
    comp = ext.find('compiler'); comp = ET.SubElement(ext, 'compiler') if comp is None else comp
    comp.set('discardvisual', 'true'); comp.set('fusestatic', 'false')
    return root


def build_model(urdf_text):
    """URDF text (from /robot_description) -> mujoco model of its joint tree."""
    import mujoco
    return mujoco.MjModel.from_xml_string(ET.tostring(_bare_urdf(urdf_text), encoding='unicode'))


def fk_poses(model, data, joints, bodies):
    """Set measured joint qpos, forward-kinematics, return {body: 4x4 pose in the model world}."""
    import mujoco
    scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for jn, val in joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0 and int(model.jnt_type[jid]) in scalar:   # int(): mjtJoint enum compare is unreliable across versions
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)
    out = {}
    for b in bodies:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            raise SystemExit(f'body {b!r} not in the live URDF; check the config frame names')
        T = np.eye(4); T[:3, :3] = data.xmat[i].reshape(3, 3); T[:3, 3] = data.xpos[i]
        out[b] = T
    return out


class Kinematics:
    """MuJoCo FK of a URDF. ``set_joints`` takes positions by joint name (a URDF mimic joint follows its leader); the
    getters then report frames in ``reference_body``. With ``tcp_body`` the model also carries the
    ``gripper/tcp_frame`` site at ``tcp_position`` / ``tcp_rotation`` in that body (GL-style axes, as the IK expects)."""

    def __init__(self, urdf_text, reference_body, tcp_body=None, tcp_rotation=None, tcp_position=(0, 0, 0)):
        import mujoco
        self.mj = mujoco
        root = _bare_urdf(urdf_text)
        self.mimic = {j.get('name'): (m.get('joint'), float(m.get('multiplier', 1)), float(m.get('offset', 0)))
                      for j in root.iter('joint') for m in j.iter('mimic')}
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
        if tcp_body is not None:
            body = spec.body(tcp_body)
            if body is None:
                raise ValueError(f'TCP body {tcp_body!r} is not in the URDF')
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.asarray(tcp_rotation, float).reshape(9))
            body.add_site(name=TCP_SITE, pos=list(tcp_position), quat=quat)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.joints = {self.model.joint(j).name: j for j in range(self.model.njnt)}
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        unsupported = [n for n, j in self.joints.items() if int(self.model.jnt_type[j]) not in scalar]
        if unsupported:
            raise ValueError(f'only hinge and slide joints are supported: {unsupported}')
        self._reference = body = self._id(mujoco.mjtObj.mjOBJ_BODY, reference_body)
        while body:                                     # the reference is the policy's world: nothing may move it
            if self.model.body_jntnum[body]:
                raise ValueError(f'reference body {reference_body!r} is moved by a joint; it must be fixed')
            body = int(self.model.body_parentid[body])
        mujoco.mj_forward(self.model, self.data)
        self._to_reference = inv_se3(self._world(self._reference))

    def _id(self, kind, name):
        index = self.mj.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f'{name!r} is not in the robot model')
        return index

    def _world(self, body):
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[body].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body]
        return T

    def check(self, bodies=(), joints=()):
        """Raise a ValueError if any name is not in the model."""
        for name in bodies:
            self._id(self.mj.mjtObj.mjOBJ_BODY, name)
        missing = [name for name in joints if name not in self.joints]
        if missing:
            raise ValueError(f'joints {missing} are not in the robot model')

    def set_joints(self, positions):
        """FK at ``positions`` (joint name -> rad or m). Every joint needs a value, given or through its mimic
        leader."""
        for name, j in self.joints.items():
            if name in positions:
                q = positions[name]
            elif name in self.mimic and self.mimic[name][0] in positions:
                leader, scale, offset = self.mimic[name]
                q = scale * positions[leader] + offset
            else:
                raise KeyError(f'no position for joint {name!r}')
            self.data.qpos[self.model.jnt_qposadr[j]] = q
        self.mj.mj_forward(self.model, self.data)
        self._to_reference = inv_se3(self._world(self._reference))

    def reference_in_world(self):
        """The reference body's pose in the model's world (the URDF root)."""
        return self._world(self._reference)

    def pose(self, body):
        """4x4 pose of a body in the reference frame."""
        return self._to_reference @ self._world(self._id(self.mj.mjtObj.mjOBJ_BODY, body))

    def site_pose(self, site=TCP_SITE):
        """4x4 pose of a site in the reference frame."""
        index = self._id(self.mj.mjtObj.mjOBJ_SITE, site)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[index].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[index]
        return self._to_reference @ T

    def joint_body(self, joint):
        """4x4 pose (reference frame) of the body a joint moves."""
        return self._to_reference @ self._world(int(self.model.jnt_bodyid[self.joints[joint]]))

    def joint_axis(self, joint):
        """(anchor, unit axis) of a joint in the reference frame."""
        j = self.joints[joint]
        R, t = self._to_reference[:3, :3], self._to_reference[:3, 3]
        return R @ self.data.xanchor[j] + t, R @ self.data.xaxis[j]

    def position(self, joint):
        return float(self.data.qpos[self.model.jnt_qposadr[self.joints[joint]]])

    def limits(self, joint):
        """(lower, upper) of a joint; infinite when the URDF leaves it unlimited."""
        j = self.joints[joint]
        if not self.model.jnt_limited[j]:
            return -math.inf, math.inf
        return float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])

    def is_hinge(self, joint):
        return int(self.model.jnt_type[self.joints[joint]]) == int(self.mj.mjtJoint.mjJNT_HINGE)
