"""The robot over ROS 2: camera and joint subscriptions, the live URDF, and, when commanding, the arm's position
command topic, the vendor's controls ownership and its homing / hardware / owner state. One node, spun on its own
thread. Camera frames are kept as the JPEG bytes that arrived and decoded only when grabbed.

Topics come from the config (the camera config: ``cameras.<name>.topic``, ``joint_state_topic``,
``robot_description_topic`` and ``motion``), with the vendor's defaults.
"""
import threading
import time
from collections import deque

import numpy as np

FRAME_MAX_AGE_S = 0.5    # a camera frame older than this is stale
JOINT_MAX_AGE_S = 0.2    # joint feedback older than this is stale: nothing moves on it


class NotReady(RuntimeError):
    """Sensor data is missing or stale."""


class RobotError(RuntimeError):
    """The vendor stack refused or did not answer (URDF, controls ownership)."""


class Ros:
    """One ROS 2 node for ``cams``. With ``command`` it also publishes the arm's position commands, switches the
    vendor's controls ownership and follows its homing, hardware and owner state."""

    def __init__(self, config, cams, command=False, name='openyam'):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import Float64MultiArray, String
        self.rclpy, self.lock = rclpy, threading.Lock()
        self._frames, self._latest = {}, {}      # camera -> (jpeg, header stamp, arrival); urdf/homing/hardware/owner
        self._joints = deque(maxlen=1024)        # (positions, header stamp, arrival), ~2 s at 500 Hz
        rclpy.init(args=[])
        self.node = node = Node(name)
        for cam in cams:
            node.create_subscription(CompressedImage, config['cameras'][cam]['topic'],
                                     lambda m, c=cam: self._put_frame(c, m), qos_profile_sensor_data)
        node.create_subscription(JointState, config.get('joint_state_topic', '/joint_states'), self._put_joints,
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, config.get('robot_description_topic', '/robot_description'),
                                 lambda m: self._put('urdf', m.data), latched)
        if command:
            from anvil_msgs.msg import ArmsResetStatus, ControlsOwner, HardwareState
            from anvil_msgs.srv import SwitchControlsOwner
            motion = config['motion']
            self._Float64MultiArray, self._SwitchOwner = Float64MultiArray, SwitchControlsOwner
            self.cmd_pub = node.create_publisher(Float64MultiArray, motion['command_topic'], 10)
            self.owner_client = node.create_client(
                SwitchControlsOwner, motion.get('owner_service', '/controls_owner_manager/switch_owner'))
            node.create_subscription(ArmsResetStatus, motion.get('homing_topic', '/arms_resetter/reset_status'),
                                     lambda m: self._put('homing', m.state), latched)
            node.create_subscription(HardwareState, motion.get('hardware_state_topic', '/hardware_state_controller/state'),
                                     lambda m: self._put('hardware', m.state),   # published on change only
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
            node.create_subscription(ControlsOwner, motion.get('owner_topic', '/controls_owner'),
                                     lambda m: self._put('owner', m.owner), latched)
        self.executor = SingleThreadedExecutor(); self.executor.add_node(node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _put(self, key, value):
        with self.lock:
            self._latest[key] = value

    def _put_frame(self, name, msg):
        with self.lock:
            self._frames[name] = (bytes(msg.data), _stamp(msg), time.monotonic())

    def _put_joints(self, msg):
        with self.lock:
            self._joints.append((dict(zip(msg.name, map(float, msg.position))), _stamp(msg), time.monotonic()))

    def wait_urdf(self, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self._latest.get('urdf'):
                    return self._latest['urdf']
            time.sleep(0.2)
        raise RobotError('robot_description not received (is robot_state_publisher up on this domain?)')

    def joints(self, max_age=JOINT_MAX_AGE_S, wait=0.0):
        """The newest joint positions (name -> rad or m), at most ``max_age`` s old; waits up to ``wait`` s for them."""
        deadline = time.monotonic() + wait
        while True:
            with self.lock:
                latest = self._joints[-1] if self._joints else None
            if latest is not None and time.monotonic() - latest[2] <= max_age:
                return dict(latest[0])
            if time.monotonic() >= deadline:
                raise NotReady('no fresh /joint_states')
            time.sleep(0.05)

    def grab(self, cams, since=0.0, wait=0.0):
        """(frames, joints): each camera's newest frame (RGB), every one arrived after ``since`` (time.monotonic) and at
        most FRAME_MAX_AGE_S old, and the newest joints; waits up to ``wait`` s. NotReady names what is missing."""
        deadline = time.monotonic() + wait
        while True:
            now = time.monotonic()
            with self.lock:
                frames = {c: self._frames.get(c) for c in cams}
                joints = self._joints[-1] if self._joints else None
            missing = [c for c, f in frames.items() if f is None or f[2] <= since or now - f[2] > FRAME_MAX_AGE_S]
            if joints is None or now - joints[2] > JOINT_MAX_AGE_S:
                missing.append('joints')
            if not missing:
                break
            if now >= deadline:
                raise NotReady(f'no fresh data from {missing}')
            time.sleep(0.02)
        return {c: _decode(c, jpeg) for c, (jpeg, _, _) in frames.items()}, dict(joints[0])

    def snapshot(self, cams, max_age=FRAME_MAX_AGE_S, max_skew=0.1):
        """(joints, frames): each camera's newest frame (RGB) and the joint sample closest to their capture time. NotReady
        if any of it arrived more than ``max_age`` s ago, or the header stamps are more than ``max_skew`` s apart."""
        with self.lock:
            missing = [c for c in cams if c not in self._frames] + ([] if self._joints else ['joints'])
            if missing:
                raise NotReady(f'no data yet from {missing}')
            frames = {c: self._frames[c] for c in cams}
            capture = float(np.mean([stamp for _, stamp, _ in frames.values()])) if frames else None
            joints = self._joints[-1] if capture is None else min(self._joints, key=lambda s: abs(s[1] - capture))
        now = time.monotonic()
        stale = [c for c, (_, _, t) in frames.items() if now - t > max_age] + (['joints'] if now - joints[2] > max_age else [])
        if stale:
            raise NotReady(f'stale data from {stale}')
        stamps = [stamp for _, stamp, _ in frames.values()] + [joints[1]]
        if max(stamps) - min(stamps) > max_skew:
            raise NotReady(f'camera and joint stamps are {max(stamps) - min(stamps):.3f} s apart')
        return dict(joints[0]), {c: _decode(c, jpeg) for c, (jpeg, _, _) in frames.items()}

    def status(self):
        """The vendor's ``homing``, ``hardware`` and controls ``owner`` state as last published (None if never seen)."""
        with self.lock:
            return {key: self._latest.get(key) for key in ('homing', 'hardware', 'owner')}

    def wait_status(self, key, timeout=3.0):
        deadline = time.monotonic() + timeout
        while self.status()[key] is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.status()[key]

    def publish_command(self, positions):
        self.cmd_pub.publish(self._Float64MultiArray(data=[float(x) for x in positions]))

    def _switch_owner(self, requester, release, timeout=10.0):
        if not self.owner_client.wait_for_service(timeout_sec=timeout):
            raise RobotError('controls-owner service unavailable (/controls_owner_manager/switch_owner)')
        future = self.owner_client.call_async(self._SwitchOwner.Request(requester=requester, release=release))
        end = time.time() + timeout
        while not future.done() and time.time() < end:
            time.sleep(0.05)
        return future

    def claim(self, requester):
        future = self._switch_owner(requester, False)
        if not future.done():
            def release_late(done):                  # a claim that succeeds after we gave up must not stay held
                if not done.cancelled() and done.result().success:
                    self.owner_client.call_async(self._SwitchOwner.Request(requester=requester, release=True))
            future.add_done_callback(release_late)
            raise RobotError('controls-owner call timed out')
        reply = future.result()
        if not reply.success:
            raise RobotError(f'could not claim controls as {requester!r}: {reply.message} '
                             f'(current owner {reply.current_owner!r})')
        print(f'claimed controls ownership as {requester!r}', flush=True)

    def release(self, requester):
        try:
            future = self._switch_owner(requester, True)
            print(f'released controls ownership ({requester!r})' if future.done() and future.result().success
                  else f'warning: releasing controls ownership ({requester!r}) was not confirmed', flush=True)
        except Exception as e:
            print(f'warning: release failed: {e}', flush=True)

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def _stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def _decode(cam, jpeg):
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise NotReady(f'a {cam} frame did not decode')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
