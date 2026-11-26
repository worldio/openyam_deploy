"""The real arm as the policy's embodiment: the observation OmniAct's model reads each step (the Rollout wire's
tensor map), with the same fields and conventions as omniact-molmospaces' eval_server (molmo_spaces/rollout/session.py
``_observation``, embodiment_gen/representations.py):

  views          (V, H, W, 3) uint8  the calibrated views (views.py), 256 px square: the client's crop-resize to
                                     256 leaves them as they are
  node_geometry  (N, 2, V, 3) f32    per node [centre, axis tip] in each view's camera frame (WODS: x right, y up,
                                     z forward), metres. A joint's centre is its anchor, or ``centre_offset`` in the
                                     body it moves; axis tip = centre + ``axis_length`` along the joint axis (the TCP:
                                     along its approach); a terminal repeats its centre
  node_proprio   (N,) f32            joint position - ``proprio_center`` (default: the midpoint of its range), rad / m;
                                     0 for the TCP and terminals
  node_pose      (N, V, 3, 4) f32    the TCP frame (WODS axes) in each camera frame; zero for every other node
  c2w            (V, 3, 4) f32       camera -> the arm base (WODS camera axes), so c2w . node_pose = the TCP in the base
  edges (E, 2) int64, axis_mask (N,) bool, joint_type (N,) int64 (0 terminal, 1 revolute, 2 prismatic, 3 TCP)
  node_names, camera_names (the V order), prompt

The node graph (configs/openyam_serve.json) is the training's linear_4310 layout on the real arm (TCP at the fingertip
plane, pad terminals, drivers at the finger carriages).
"""
import numpy as np

from .kinematics import Kinematics
from .views import View

JOINT_TERMINAL, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_TCP = 0, 1, 2, 3
WODS_TCP = np.diag([1.0, 1.0, -1.0, 1.0])       # the GL-style TCP site -> its WODS axes (geometry.GL_TO_WODS)


class Embodiment:
    """The node graph (``graph``) on FK of the live URDF, and the calibrated views of ``cameras`` (the calibrated
    camera config's entries). ``build`` turns measured joints and raw camera frames into the observation."""

    def __init__(self, graph, cameras, urdf_text, reference_body):
        self.nodes = graph['nodes']
        self.names = [node['name'] for node in self.nodes]
        tcp = [node for node in self.nodes if node.get('tcp')]
        if len(tcp) != 1 or len(set(self.names)) != len(self.names):
            raise ValueError('the node graph needs unique node names and exactly one TCP node')
        self.kin = Kinematics(urdf_text, reference_body, tcp[0]['body'], tcp[0]['site_rotation'],
                              tcp[0].get('site_position', (0, 0, 0)))
        self.kin.check(bodies=[n['body'] for n in self.nodes if 'body' in n],
                       joints=[n['joint'] for n in self.nodes if 'joint' in n])
        self.views = {name: View(name, entry) for name, entry in cameras.items()}
        for view in self.views.values():
            self.kin.check(bodies=[view.parent])
        self.axis_length = float(graph['axis_length'])
        types, centers = [], []
        for node in self.nodes:
            if 'joint' in node:
                lo, hi = self.kin.limits(node['joint'])
                types.append(JOINT_REVOLUTE if self.kin.is_hinge(node['joint']) else JOINT_PRISMATIC)
                centers.append(node.get('proprio_center', (lo + hi) / 2.0 if np.isfinite(lo + hi) else 0.0))
            else:
                types.append(JOINT_TCP if node.get('tcp') else JOINT_TERMINAL)
                centers.append(0.0)
        self.joint_type = np.asarray(types, np.int64)
        self.axis_mask = np.isin(self.joint_type, (JOINT_REVOLUTE, JOINT_PRISMATIC))
        self.tcp_node = int(np.flatnonzero(self.joint_type == JOINT_TCP)[0])
        self.center = np.asarray(centers)
        index = {name: i for i, name in enumerate(self.names)}
        self.edges = np.asarray([[index[a], index[b]] for a, b in graph['edges']], np.int64).reshape(-1, 2)

    def tcp(self):
        """The TCP frame (WODS axes) in the reference body at the kinematics' current joints, 4x4."""
        return self.kin.site_pose() @ WODS_TCP

    def geometry(self, joints):
        """At ``joints`` (name -> position): per node [centre, axis tip] in the reference frame (N, 2, 3), the node
        proprio (N,) and the TCP (4x4)."""
        kin, length = self.kin, self.axis_length
        kin.set_joints(joints)
        points = np.zeros((len(self.nodes), 2, 3))
        proprio = np.zeros(len(self.nodes))
        tcp = self.tcp()
        for i, node in enumerate(self.nodes):
            if 'joint' in node:
                centre, axis = kin.joint_axis(node['joint'])
                if 'centre_offset' in node:
                    centre = (kin.joint_body(node['joint']) @ np.r_[node['centre_offset'], 1.0])[:3]
                points[i] = centre, centre + length * axis
                proprio[i] = kin.position(node['joint']) - self.center[i]
            elif node.get('tcp'):
                points[i] = tcp[:3, 3], tcp[:3, 3] + length * tcp[:3, 2]        # WODS z: the approach
            else:
                points[i] = (kin.pose(node['body']) @ np.r_[node['point'], 1.0])[:3]
        return points, proprio, tcp

    def build(self, joints, frames, view_names, prompt):
        """The observation for measured ``joints`` (name -> position) and raw RGB ``frames`` (camera name -> image),
        the views in ``view_names`` order."""
        points, proprio, tcp = self.geometry(joints)
        images, geometry, poses, c2w = [], [], [], []
        for name in view_names:
            view = self.views[name]
            camera = view.c2w(self.kin)                                # view -> reference, WODS axes
            pose = np.zeros((len(self.nodes), 3, 4))
            pose[self.tcp_node] = (np.linalg.inv(camera) @ tcp)[:3]
            images.append(view.render(frames[name]))
            geometry.append((points - camera[:3, 3]) @ camera[:3, :3])
            poses.append(pose)
            c2w.append(camera[:3])
        return {
            'views': np.stack(images),
            'node_geometry': np.stack(geometry, axis=2).astype(np.float32),
            'node_proprio': proprio.astype(np.float32),
            'node_pose': np.stack(poses, axis=1).astype(np.float32),
            'c2w': np.stack(c2w).astype(np.float32),
            'edges': self.edges.copy(),
            'axis_mask': self.axis_mask.copy(),
            'joint_type': self.joint_type.copy(),
            'node_names': list(self.names),
            'camera_names': list(view_names),
            'prompt': prompt,
        }
