"""Moving the arm: position setpoints streamed to the vendor's forward position controller.

``move`` is a point-to-point move (smoothstep, zero velocity at both ends, within a speed and acceleration limit):
the calibration jog and the server's reset. ``stream`` plays one policy tick: a straight ramp from the last command to
the new target over the tick, so consecutive ticks join into one continuous motion. Both check before every setpoint
(``check``): a stop request, the vendor's hardware not paused or e-stopped, fresh joint feedback and, given an
``owner``, the controls still ours. ``Controls`` holds the vendor's controls ownership for a block.
"""
import math
import time

import numpy as np

from .ros import JOINT_MAX_AGE_S, NotReady

SETPOINT_HZ = 250                    # setpoint publish rate: fine steps (0.18 deg at the move speed limit); the vendor
                                     # controller holds each until the next, and a stiff wrist (kp 100) shook on the
                                     # 1.5 deg steps of 30 Hz
SLEW_RAD_S = math.radians(45.0)      # move: speed limit
ACCEL_RAD_S2 = math.radians(90.0)    # move: acceleration limit (smoothstep: zero velocity at both ends)
MIN_MOVE_S = 0.4                     # move: shortest move


class Stopped(RuntimeError):
    """Motion stopped early: on request (Ctrl-C, SIGTERM, the client gone), the hardware paused or e-stopped, stale joint
    feedback, or the controls taken; the arm holds where it is."""


def check(ros, stop, owner=None):
    """Raise Stopped unless the motion may go on: no stop request (``stop``, a threading.Event), the hardware not paused
    or e-stopped, fresh joint feedback, and (``owner``) the controls still held by that requester."""
    if stop.is_set():
        raise Stopped('stop requested')
    status = ros.status()
    if status['hardware'] not in (None, 'active'):
        raise Stopped(f"hardware state is {status['hardware']!r}")
    if owner is not None and status.get('owner') not in (None, owner):
        raise Stopped(f"the controls owner is now {status['owner']!r}")
    try:
        ros.joints(JOINT_MAX_AGE_S)
    except NotReady as error:
        raise Stopped(str(error)) from None


def move(ros, start, target, stop, owner=None):
    """Stream setpoints from ``start`` to ``target`` (command-joint order; the gripper last) at SETPOINT_HZ,
    smoothstep with zero velocity at both ends, within the speed and acceleration limits of the arm joints, checking
    before every setpoint. Returns ``target``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    span = float(np.abs(target - start)[:-1].max())
    duration = max(MIN_MOVE_S, 1.5 * span / SLEW_RAD_S, math.sqrt(6.0 * span / ACCEL_RAD_S2))
    n = max(1, int(math.ceil(duration * SETPOINT_HZ)))
    t0 = time.monotonic()
    for k in range(1, n + 1):
        check(ros, stop, owner)
        s = k / n
        ros.publish_command(start + (3 * s ** 2 - 2 * s ** 3) * (target - start))
        time.sleep(max(0.0, t0 + k / SETPOINT_HZ - time.monotonic()))    # on a clock: the checks' time doesn't add up
    return target


def stream(ros, start, target, duration, stop, owner=None):
    """Stream setpoints on the straight line from ``start`` to ``target`` over ``duration`` s at SETPOINT_HZ (one
    policy tick), checking before every setpoint. Returns ``target``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    n = max(1, int(round(duration * SETPOINT_HZ)))
    t0 = time.monotonic()
    for k in range(1, n + 1):
        check(ros, stop, owner)
        ros.publish_command(start + (k / n) * (target - start))
        time.sleep(max(0.0, t0 + k / SETPOINT_HZ - time.monotonic()))
    return target


def hold(ros, names):
    """Command the measured position (best effort), as the vendor's own shutdown does."""
    try:
        joints = ros.joints(JOINT_MAX_AGE_S)
        ros.publish_command([joints[n] for n in names])
    except Exception:
        pass


class Controls:
    """``with Controls(ros, requester):`` the vendor's controls, held for the block: the arm must be homed, and the
    claim is given back on exit (the drives ignore commands from anyone but the owner)."""

    def __init__(self, ros, requester):
        self.ros, self.requester = ros, requester

    def __enter__(self):
        homing = self.ros.wait_status('homing')
        if homing != 'homed':
            raise Stopped(f'the arm is not homed (reset status: {homing!r})')
        self.ros.claim(self.requester)
        return self

    def __exit__(self, *exc):
        self.ros.release(self.requester)
