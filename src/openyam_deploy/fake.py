"""A simulated robot with the Ros interface (ros.py): the joints follow each command at once (or as ``physics`` says:
physics(command, measured) -> measured, both arrays in command-joint order), the cameras send noise frames at their
lens's resolution, the vendor state is homed and active. serve.py --fake and the tests run on it."""
import threading

import numpy as np


class FakeRobot:
    def __init__(self, config, cams, urdf_text, joints, physics=None):
        self.urdf, self.names = urdf_text, list(config['motion']['command_joints'])
        self.sizes = {c: tuple(config['cameras'][c]['raw']['size']) for c in cams}
        self.q = dict(joints)                              # joint name -> position
        self.owner, self.commands, self.lock, self.physics = None, 0, threading.Lock(), physics
        self._rng = np.random.default_rng(0)

    def wait_urdf(self, timeout=15.0):
        return self.urdf

    def joints(self, max_age=0.2, wait=0.0):
        with self.lock:
            return dict(self.q)

    def snapshot(self, cams, max_age=0.5, max_skew=0.1):
        frames = {c: self._rng.integers(0, 255, (self.sizes[c][1], self.sizes[c][0], 3), np.uint8) for c in cams}
        return self.joints(), frames

    def grab(self, cams, since=0.0, wait=0.0):
        joints, frames = self.snapshot(cams)
        return frames, joints

    def status(self):
        return {'homing': 'homed', 'hardware': 'active', 'owner': self.owner}

    def wait_status(self, key, timeout=3.0):
        return self.status()[key]

    def publish_command(self, positions):
        with self.lock:
            if self.physics is not None:
                positions = self.physics(np.asarray(positions, float), np.array([self.q[n] for n in self.names]))
            self.q.update(zip(self.names, map(float, positions)))
            self.commands += 1

    def claim(self, requester):
        self.owner = requester

    def release(self, requester):
        self.owner = None

    def close(self):
        pass



def arm_physics(conv):
    """A FakeRobot ``physics`` for the converter ``conv``'s arm: it sags from its command back to its target (the
    gravity offset undone), as the real arm does."""
    return lambda command, measured: command - conv.gravity_offset(command)
