"""One episode on the real robot, as omniact-molmospaces' eval_server runs one in simulation
(molmo_spaces/rollout/session.py), where the two differ only because a real robot must:

* Reset: nothing rebuilds the scene; the arm moves to the start pose (``start_pose_deg``, its base joint turned and
  its gripper moved out as the arm's ``start_turns`` say for the prompt: ``start_pose``) and the gripper closes, then
  the first observation. The controls are held from the first Reset to Close.
* Step: the step's targets (actions.py) plus the gravity offset, streamed over one policy tick (``tick_s``, the
  training's 66 ms) at 250 Hz; then the observation. The episode is done after ``task_horizon`` steps, or on an IK
  failure when the client opened with continue_on_ik_failure=0. Nothing judges success on a real robot, so ``info``
  carries no "success" (the client counts that as a failure; judge the recorded rollouts).
* ``motion`` False (serve --no-motion): the same loop, the arm never commanded.

Every step is logged (``log_dir``/steps.jsonl) from a background thread, so the log never slows the loop. With a
``recording`` (record.py), each episode is also recorded when it ends: a WODS item and an annotated video.
"""
import json
import math
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import motion
from .actions import Converter, degrees, tcp_pose
from .embodiment import Embodiment
from .record import Episode
from .ros import JOINT_MAX_AGE_S, NotReady

SNAPSHOT_TRIES, SNAPSHOT_WAIT_S = 20, 0.05       # ride out a late camera frame or two


class Session:
    n_episodes = 1

    def __init__(self, ros, embodiment, converter, settings, *, prompt, views, task_horizon, continue_on_ik_failure,
                 seed=0, motion_on=True, requester='diagnostics', log_dir=None, recording=None):
        self.ros, self.emb, self.conv, self.settings = ros, embodiment, converter, settings
        self.prompt, self.views, self.task_horizon = prompt, sorted(views), int(task_horizon)
        self.start_deg, self.start_turn, self.start_move = start_pose(settings, prompt, embodiment, converter.names)
        self.continue_on_ik_failure, self.seed = bool(continue_on_ik_failure), int(seed)
        self.motion_on, self.requester = motion_on, requester
        self.names = converter.names
        self.stop_event, self.controls, self.command = threading.Event(), None, None
        self.steps, self.obs, self.stopped, self._replied, self._joints = 0, None, False, None, None
        self._log = _Log(log_dir) if log_dir else None
        self.recording = recording if self._log else None
        self._episode, self._episodes = None, 0
        if self._log:
            self._log.write('session.json', {'prompt': prompt, 'views': self.views, 'task_horizon': self.task_horizon,
                                             'continue_on_ik_failure': self.continue_on_ik_failure, 'seed': self.seed,
                                             'motion': motion_on, 'start_pose_deg': self.start_deg.tolist(),
                                             'start_turn_deg': self.start_turn, 'start_move_tcp_m': self.start_move,
                                             'started': time.strftime('%Y-%m-%d %H:%M:%S')})
            if self.recording:
                self._log.call(self.recording.begin, self._log.dir)

    # ---- the protocol
    def reset(self, episode_idx):
        if episode_idx != 0:
            raise ValueError(f'the robot serves one episode, index 0 (not {episode_idx})')
        self._finish('the client reset')
        self.stop_event.clear()
        if self.motion_on:
            if self.controls is None:
                self.controls = motion.Controls(self.ros, self.requester).__enter__()
            measured = self._measured()
            start = np.r_[np.radians(self.start_deg), measured[-1]]
            self.command = motion.move(self.ros, measured, self._with_gravity(start), self.stop_event, self.requester)
            closed = np.r_[start[:-1], float(self.settings['reset_gripper_m'])]   # start as training does: closed
            self.command = motion.move(self.ros, self.command, self._with_gravity(closed), self.stop_event, self.requester)
        self.conv.reset(self.seed)
        self.steps = 0
        obs = self._observe()
        if self.recording:
            self._episode = Episode(self._episodes, self.prompt, self.names, self.settings['tick_s'], self.motion_on)
            self._episodes += 1
            self._episode.frame(obs, self._joints)
        return obs

    def step(self, action, node_action_pose, action_mask, node_action_pose_mask):
        if self.obs is None:
            raise ValueError('Reset before Step')
        arrived = time.monotonic()
        timing = {'client_s': None if self._replied is None else round(arrived - self._replied, 4)}
        record = {'step': self.steps, 'action': np.asarray(action).tolist(),
                  'action_mask': np.asarray(action_mask).tolist(),
                  'pose_mask': np.asarray(node_action_pose_mask).tolist(),
                  'tcp_target': np.asarray(node_action_pose)[self.emb.tcp_node].tolist(), 'timing': timing}
        try:
            t = time.monotonic()
            start, target, info = self.conv.convert(self.ros.joints(JOINT_MAX_AGE_S), action, node_action_pose,
                                                    action_mask, node_action_pose_mask)
            record.update(start=start.tolist(), target=target.tolist(), **info)
            timing['convert_s'] = round(time.monotonic() - t, 4)
            if self.motion_on:
                t = time.monotonic()
                command = self._with_gravity(target)
                record['gravity_offset_deg'] = degrees((command - target)[:-1])
                self.command = motion.stream(self.ros, start if self.command is None else self.command, command,
                                             float(self.settings['tick_s']), self.stop_event, self.requester)
                timing['move_s'] = round(time.monotonic() - t, 4)
            t = time.monotonic()
            obs = self._observe()
            timing['observe_s'] = round(time.monotonic() - t, 4)
            record['tcp_reached'] = tcp_pose(self.emb, self._joints).tolist()
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            self._append(record)
            if isinstance(error, motion.Stopped):
                motion.hold(self.ros, self.names)
            if 'target' in record and self._episode:           # a step that commanded the arm ends the episode
                self._episode.step(record)
                self._finish(record['error'])
            raise
        self.steps += 1
        ik_failed = bool(info.get('ik_failed'))
        done = self.steps >= self.task_horizon or (ik_failed and not self.continue_on_ik_failure)
        result = {'step': self.steps}
        for key in ('ik_outcome', 'ik_pos_err', 'ik_rot_err', 'ik_step'):
            if key in info:
                result[key] = info[key]
        timing['server_s'] = round(time.monotonic() - arrived, 4)
        record['done'] = done
        self._append(record)
        if self._episode:
            self._episode.step(record)
            self._episode.frame(obs, self._joints)
            if done:
                self._finish('the task horizon' if self.steps >= self.task_horizon else 'an IK failure')
        self._replied = time.monotonic()
        return obs, done, result

    def stop(self):
        """Stop any motion at its next setpoint (safe from any thread: the client went away, Close, shutdown)."""
        self.stopped = True
        self.stop_event.set()

    def close(self, end='the client closed'):
        """Hold the arm, give the controls back, and end the episode being recorded (``end``: why)."""
        self.stop()
        if self.controls is not None:
            motion.hold(self.ros, self.names)
            self.controls, controls = None, self.controls
            controls.__exit__(None, None, None)                  # give the controls back
        self._finish(end)
        if self._log:
            self._log.close()

    # ---- helpers
    def _measured(self):
        joints = self.ros.joints(JOINT_MAX_AGE_S, wait=1.0)
        return np.array([joints[n] for n in self.names], float)

    def _with_gravity(self, target):
        return target + self.conv.gravity_offset(target)

    def _observe(self):
        cameras = self.views
        for attempt in range(SNAPSHOT_TRIES):
            try:
                self._joints, frames = self.ros.snapshot(cameras)
                break
            except NotReady:
                if attempt == SNAPSHOT_TRIES - 1:
                    raise
                time.sleep(SNAPSHOT_WAIT_S)
        self.obs = self.emb.build(self._joints, frames, self.views, self.prompt)
        return self.obs

    def _append(self, record):
        if self._log:
            self._log.append('steps.jsonl', record)

    def _finish(self, end):
        """End the episode being recorded, if any, and hand it to the recorder (on the log thread)."""
        episode, self._episode = self._episode, None
        if episode is not None:
            episode.end = end
            self._log.call(self.recording.finish, episode, self._log.dir)


def setup(config, settings, urdf_text, gravity=True):
    """What serving needs, from the calibrated camera config and the serve settings (configs/openyam_serve.json):
    ``emb`` (the embodiment: node graph, FK, the calibrated views), ``conv`` (the action converter, with the gravity
    offset unless ``gravity`` is off) and ``views`` (the calibrated cameras, sorted)."""
    cameras = {name: entry for name, entry in config['cameras'].items() if 'calibration' in entry}
    if not cameras:
        raise SystemExit('no calibrated camera in the config: run the calibration first (README.md, Calibrate)')
    emb = Embodiment(settings['node_graph'], cameras, urdf_text, config['reference_body'])
    conv = Converter(emb, config['motion']['command_joints'], settings['gravity'] if gravity else None)
    return SimpleNamespace(emb=emb, conv=conv, views=sorted(cameras))


START_MOVE_TOL = (0.005, 5.0)     # a start_turns "move_tcp" must get the TCP within this (m) of its target with the
                                  # gripper turned at most this (deg), or Reset refuses


def start_pose(settings, prompt, embodiment=None, names=None):
    """The arm's start pose for ``prompt``: the settings' ``start_pose_deg`` (the arm joints, deg), changed by the first
    of the settings' ``start_turns`` whose words the prompt contains (matched anywhere in the prompt, case ignored):
    [{"prompt_words": ["open", "close"], "turn_deg": -30, "move_tcp": [0.10, 0.0]}, ...]. "turn_deg" turns the first
    joint, the base swivel, positive counterclockwise seen from above (to the arm's left); "move_tcp" (optional) then
    moves the TCP that far (m) along where the gripper points, horizontally, and up, the gripper kept as it was
    (_moved_tcp, through ``embodiment``'s kinematics; ``names``: the command joints). Returns (the pose, the turn in
    deg, the move or None)."""
    pose = np.array(settings['start_pose_deg'], float)
    text = (prompt or '').lower()
    for rule in settings.get('start_turns', []):
        if any(str(word).lower() in text for word in rule.get('prompt_words', [])):
            pose[0] += float(rule['turn_deg'])
            move = rule.get('move_tcp')
            if move and any(move):
                pose = _moved_tcp(embodiment, names, pose, float(move[0]), float(move[1]))
            return pose, float(rule['turn_deg']), (None if not move else [float(move[0]), float(move[1])])
    return pose, 0.0, None


def _moved_tcp(embodiment, names, pose_deg, out, up):
    """The arm joints (deg) that move the TCP ``out`` m along where the gripper points, horizontally, and ``up`` m
    from its pose at ``pose_deg``, the gripper turned as little as it can: solved on the embodiment's kinematics within
    the joint limits (5 deg in). ValueError unless the TCP gets within START_MOVE_TOL."""
    from scipy.optimize import minimize
    kin, tcp = embodiment.kin, embodiment.nodes[embodiment.tcp_node]['body']
    arm = list(names[:6])

    def pose_at(q):
        kin.set_joints({**{n: float(v) for n, v in zip(arm, q)}, names[6]: 0.0})
        return kin.pose(tcp)
    q0 = np.radians(pose_deg[:6])
    T0 = pose_at(q0)
    forward = np.r_[T0[:2, 2], 0.0]                       # the gripper's axis (TCP z), horizontal
    forward /= max(np.linalg.norm(forward), 1e-9)
    target = T0[:3, 3] + out * forward + (0.0, 0.0, up)

    def cost(q):                                          # 2 mm off target costs as much as ~3 deg of gripper turn
        T = pose_at(q)
        return (np.linalg.norm(T[:3, 3] - target) / 0.002) ** 2 + (np.linalg.norm(T[:3, :3] - T0[:3, :3]) / 0.05) ** 2
    margin = np.radians(5.0)
    bounds = [(lo + margin, hi - margin) for lo, hi in (kin.limits(n) for n in arm)]
    q = minimize(cost, q0, method='L-BFGS-B', bounds=bounds).x
    T = pose_at(q)
    miss = float(np.linalg.norm(T[:3, 3] - target))
    turned = math.degrees(float(np.linalg.norm(_log_so3(T0[:3, :3].T @ T[:3, :3]))))
    if miss > START_MOVE_TOL[0] or turned > START_MOVE_TOL[1]:
        raise ValueError(f'start_turns move_tcp [{out:g}, {up:g}]: the arm can only get the TCP within '
                         f'{1000 * miss:.0f} mm of it with the gripper turned {turned:.0f} deg (limits '
                         f'{1000 * START_MOVE_TOL[0]:.0f} mm, {START_MOVE_TOL[1]:.0f} deg): ask for less')
    return np.degrees(q)


def _log_so3(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, float))[0].reshape(3)


def open_options(bundle, options, served, default_horizon):
    """An Open's bundle and options -> (prompt, views, task_horizon, continue_on_ik_failure, seed), by
    omniact-molmospaces' rules (molmo_spaces/rollout/replay_session.py) where a real robot allows: the prompt is the
    bundle's ``wods_meta.json`` caption, else a ``prompt`` option, else empty (OmniAct's client without --replay-root
    sends no bundle and gives the policy its --prompt itself); ``cameras`` is a ";"-separated subset of the ``served``
    views; ``task_horizon`` and ``seed`` are ints; ``continue_on_ik_failure`` and ``end_on_success`` are flags (1/0,
    true/false, yes/no, on/off), the first default on and the second moot (nothing judges success here). The rest of
    the bundle, the simulated scene, has no use on a real robot."""
    try:
        caption = json.loads(bundle.get('wods_meta.json') or b'{}').get('caption')
    except (ValueError, AttributeError):
        caption = None
    if not isinstance(caption, str) or not caption.strip():
        caption = str(options.get('prompt') or '')
    views = sorted(served)
    if options.get('cameras'):
        views = [c for c in options['cameras'].split(';') if c]
        unknown = sorted(set(views) - set(served))
        if unknown:
            raise ValueError(f'cameras {unknown} are not served (served: {sorted(served)})')
    horizon = int(options['task_horizon']) if options.get('task_horizon') else int(default_horizon)
    seed = int(options['seed']) if options.get('seed') else 0
    for key in ('continue_on_ik_failure', 'end_on_success'):
        if key in options:
            _flag(key, options[key])
    continue_on_ik_failure = _flag('continue_on_ik_failure', options.get('continue_on_ik_failure', '1'))
    return caption, views, horizon, continue_on_ik_failure, seed


def _flag(key, value):
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'option {key}={value!r}: expected 1/0, true/false, yes/no or on/off')


class _Log:
    """A run directory written from a background thread."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, name, value):
        self._queue.put(('w', name, value))

    def append(self, name, value):
        self._queue.put(('a', name, value))

    def call(self, function, *args):
        self._queue.put(('call', function, args))

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=60.0)                   # an episode's save takes a second or two

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            mode, name, value = item
            try:
                if mode == 'call':
                    name(*value)
                    continue
                with open(self.dir / name, mode) as f:
                    f.write(json.dumps(value, default=_plain) + ('\n' if mode == 'a' else ''))
            except Exception as error:                   # a failed write loses that write, never the thread
                print(f'log: {type(error).__name__}: {error}', flush=True)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
