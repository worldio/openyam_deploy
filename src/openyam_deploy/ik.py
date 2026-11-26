"""Inverse kinematics for the pose form of an action.

An ``AbstractAction`` may carry ``node_action_pose``: the commanded TCP as an SE(3) ``[R | t]``, house-world
<- WODS-style TCP axes (x toward the right pad, y up, z forward; det -1, exactly the wods label). The simulated
robot takes joint-position targets, and ``SemanticAction.to_env`` resolves that pose into arm targets
through ``TcpIK``, built once per episode from the embodiment's FK model (``Kinematics``: the assembled robot
with the ``tcp_frame`` site) and the actuated arm joints:

* ``solve``: damped least squares (differential IK) warm-started from the arm's CURRENT configuration --
  local by design, so it descends from where the arm stands instead of hopping to a distant elbow-up /
  wrist-flipped branch that reaches the same pose. A globally seeded variant, picking on residual alone across
  random restarts, was rejected: it produced 263-308 deg single-step joint jumps in a 15 Hz control loop.
  Redundant degrees of freedom go to a secondary objective:
  the configuration is pulled toward a posture reference in the Jacobian's nullspace while the pose error
  is held inside the 0.1 mm / 0.1 mrad budget.
* ``fk_world``: the companion forward query, the tool frame at any configuration in the wire convention --
  what the eval uses to measure how far the policy's two heads (joint action vs pose) disagree.
* ``command``: the graduated safety policy around the solver, one call per control tick. The nullspace
  reference defaults to the live arm; ``hybrid`` mode hands the joint head's own target in instead, so the
  two heads agree on posture as well as on the gripper pose. Position error above 5 mm, orientation error
  above 5 deg, or a single-step joint motion above 10 deg triggers a second local attempt from nearby seeds,
  preferring candidates within 20 deg of the current arm -- among accurate ones the smallest joint move, and
  when none is accurate the lowest pose error (so an infeasible target is still approached as closely as the
  clamp allows); a step still above 20 deg is CLAMPED rather than
  commanded, and eight consecutive clamps are reported as kinematic THRASHING (the target is reachable
  only through another branch, so the arm would creep forever). A residual above 2 cm / 15 deg means the
  target is genuinely out of reach: reported as INFEASIBLE, an outcome distinct from a task failure
  (``infeasible_requires_jump`` raises it only when the large residual coincides with a teleporting joint
  step). These two are ``IK_FAILURES``; whether one ends the episode is the rollout session's decision.

Everything is solved in the robot's base frame on the robot-only model; the house-world target comes in
through ``base_pose`` like every other FK quantity in ``representations``. The wire rotation is WODS-style
(z forward); the site it must match is GL-style (z back), the two differing by ``GL_TO_WODS`` -- an
involution, so the same matrix converts both ways."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Vendored verbatim from omniact-molmospaces (molmo_spaces/embodiment_gen/ik.py, the IK of the eval_server this
# server stands in for); Copyright 2026 Allen Institute for AI, Apache-2.0. Only these two imports are replaced by their values
# (constants.py, representations.py); the solver and limits are unchanged.
GRIPPER_PREFIX, TCP_SITE = "gripper/", "tcp_frame"
GL_TO_WODS = np.diag([1.0, 1.0, -1.0])

# Thresholds calibrated on measured rollouts.
POSE_BUDGET = 1e-4                       # m / rad: the primary-task error the nullspace pull must keep
RETRY_POS, RETRY_ROT = 5e-3, np.radians(5.0)          # residual that triggers a second local attempt
RETRY_STEP = np.radians(10.0)                          # single-step joint motion that triggers it
CLAMP_STEP = np.radians(20.0)                          # a step beyond this is clamped, not commanded
THRASH_CLAMPS = 8                                      # consecutive clamps reported as thrashing
INFEASIBLE_POS, INFEASIBLE_ROT = 2e-2, np.radians(15.0)   # residual reported as infeasible
RETRY_SEEDS, RETRY_SPREAD = 3, np.radians(8.0)         # nearby seeds: count and per-joint perturbation
IK_FAILURES = ("thrashing", "infeasible")              # the ik_outcome values that mean the IK failed


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray          # (n_arm,) arm configuration
    pos_err: float         # m, achieved vs target
    rot_err: float         # rad


class TcpIK:
    """DLS inverse kinematics for the ``tcp_frame`` site + the per-tick safety policy. Stateful only for the
    consecutive-clamp counter; build one per episode."""

    def __init__(self, kin, *, infeasible_requires_jump: bool = False, seed: int = 0) -> None:
        self.kin, self.mj, self.model, self.data = kin, kin.mujoco, kin.model, kin.data
        self.site = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, kin.prefix + GRIPPER_PREFIX + TCP_SITE)
        if self.site < 0:
            raise ValueError("TcpIK needs the tcp_frame site (Kinematics injects it)")
        self.n = kin.n_arm                                            # arm joints = the first n joints / dofs
        assert all(int(self.model.jnt_qposadr[j]) == j == int(self.model.jnt_dofadr[j]) for j in range(self.n))
        limited = self.model.jnt_limited[:self.n].astype(bool)
        self.lo = np.where(limited, self.model.jnt_range[:self.n, 0], -np.inf)
        self.hi = np.where(limited, self.model.jnt_range[:self.n, 1], np.inf)
        self.infeasible_requires_jump = bool(infeasible_requires_jump)
        self._rng = np.random.default_rng(seed)
        self.clamps = 0                                               # consecutive clamped steps

    # ---------------------------------------------------------------- forward
    def fk(self, q, gripper) -> tuple[np.ndarray, np.ndarray]:
        """The site at ``q`` (gripper as given), base-local: ``(position (3,), rotation (3, 3))`` -- the site's own
        GL-style axes."""
        self.kin.pose({"arm": q, "gripper": gripper})
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def fk_world(self, q, gripper, base_pose) -> np.ndarray:
        """The tool frame at ``q`` in the wire convention: ``(3, 4)`` house-world <- WODS-style TCP axes, comparable to
        a ``node_action_pose`` row."""
        pos, rot = self.fk(q, gripper)
        base_rot, base_pos = _base(base_pose)
        return np.concatenate([base_rot @ rot @ GL_TO_WODS, (base_rot @ pos + base_pos)[:, None]], axis=1)

    # ---------------------------------------------------------------- solver
    def solve(self, target_pos, target_rot, seed_q, posture, gripper, *, iters: int = 100,
              damping: float = 0.05, posture_gain: float = 0.2, posture_rounds: int = 3) -> IKResult:
        """Damped least squares from ``seed_q`` (base-local target, the site's axes). The tool pose is the primary
        task; the pull toward ``posture`` is strictly secondary: after the task converges, each round takes a
        nullspace step toward the posture and re-projects onto the task, and is kept only while the pose stays
        inside ``POSE_BUDGET`` (a damped pseudo-inverse is not an exact projector, and active joint limits leak
        the secondary term into the task). Joint limits clamped every iteration. Returns the configuration and
        the pose error it actually achieved."""
        q = self._descend(seed_q, target_pos, target_rot, gripper, None, iters, damping, 0.0)
        if posture is not None and self._within_budget(q, target_pos, target_rot, gripper):
            for _ in range(posture_rounds):
                trial = self._descend(q, target_pos, target_rot, gripper, posture, iters // 2, damping, posture_gain)
                trial = self._descend(trial, target_pos, target_rot, gripper, None, iters // 2, damping, 0.0)
                if not self._within_budget(trial, target_pos, target_rot, gripper):
                    break
                q = trial
        pos, rot = self.fk(q, gripper)
        return IKResult(q, float(np.linalg.norm(target_pos - pos)), float(np.linalg.norm(_rotation_error(rot, target_rot))))

    def _descend(self, q, target_pos, target_rot, gripper, posture, iters, damping, gain) -> np.ndarray:
        """DLS iterations on the task error; with ``posture`` also the nullspace pull toward it."""
        q = np.asarray(q, np.float64).copy()
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        eye_n, eye6 = np.eye(self.n), np.eye(6)
        for _ in range(iters):
            pos, rot = self.fk(q, gripper)
            err = np.concatenate([target_pos - pos, _rotation_error(rot, target_rot)])
            self.mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, :self.n], jacr[:, :self.n]])
            pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * eye6, eye6)   # damped pseudo-inverse
            dq = pinv @ err
            if posture is not None:
                dq = dq + (eye_n - pinv @ jac) @ (gain * (posture - q))
            if np.linalg.norm(err[:3]) < POSE_BUDGET and np.linalg.norm(err[3:]) < POSE_BUDGET \
                    and np.abs(dq).max() < 1e-6:
                break
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def _within_budget(self, q, target_pos, target_rot, gripper) -> bool:
        pos, rot = self.fk(q, gripper)
        return bool(np.linalg.norm(target_pos - pos) < POSE_BUDGET
                    and np.linalg.norm(_rotation_error(rot, target_rot)) < POSE_BUDGET)

    # ---------------------------------------------------------------- per-tick policy
    def command(self, pose, arm, gripper, base_pose, *, posture=None) -> tuple[np.ndarray, dict]:
        """One control tick: the arm joint targets for an absolute ``pose`` -- ``(3, 4)`` house-world <- WODS-style
        TCP axes, a ``node_action_pose`` row -- from the live ``arm``, plus ``info`` (``ik_outcome`` ok / retry /
        clamped / thrashing / infeasible, ``ik_pos_err`` m, ``ik_rot_err`` rad, ``ik_step`` rad = the largest joint
        move commanded; ``IK_FAILURES`` are the failed outcomes). ``posture``, the nullspace
        reference, defaults to the live ``arm``; hybrid mode passes the joint head's target. The descent always
        starts from the live arm."""
        target_pos, target_rot = _target_in_base(np.asarray(pose, np.float64).reshape(3, 4), base_pose)
        arm = np.asarray(arm, np.float64)
        posture = arm if posture is None else np.asarray(posture, np.float64)
        step_of = lambda r: float(np.abs(r.q - arm).max())
        best = self.solve(target_pos, target_rot, arm, posture, gripper)
        outcome = "ok"
        if best.pos_err > RETRY_POS or best.rot_err > RETRY_ROT or step_of(best) > RETRY_STEP:
            outcome = "retry"                                          # a second local attempt, nearby seeds
            candidates = [best] + [
                self.solve(target_pos, target_rot,
                           np.clip(arm + self._rng.normal(0.0, RETRY_SPREAD, self.n), self.lo, self.hi),
                           posture, gripper)
                for _ in range(RETRY_SEEDS)]
            accurate = [c for c in candidates if c.pos_err <= RETRY_POS and c.rot_err <= RETRY_ROT]
            pool = accurate or candidates
            near = [c for c in pool if step_of(c) <= CLAMP_STEP]
            error_of = lambda c: (c.pos_err, c.rot_err)
            # accurate: the smallest joint move (stay on the branch); none accurate: the closest pose within the clamp
            best = min(near, key=step_of if accurate else error_of) if near else min(pool, key=error_of)
        jump = step_of(best) > CLAMP_STEP
        q = best.q
        if jump:                                                       # clamp the motion, never teleport
            q = arm + (best.q - arm) * (CLAMP_STEP / step_of(best))
            self.clamps += 1
            outcome = "clamped"
        else:
            self.clamps = 0
        if self.clamps >= THRASH_CLAMPS:
            outcome = "thrashing"
        infeasible = best.pos_err > INFEASIBLE_POS or best.rot_err > INFEASIBLE_ROT
        if infeasible and (jump or not self.infeasible_requires_jump):
            outcome = "infeasible"
        info = {"ik_outcome": outcome, "ik_pos_err": best.pos_err, "ik_rot_err": best.rot_err,
                "ik_step": float(np.abs(q - arm).max())}
        return q, info


# -------------------------------------------------------------------- helpers
def pose_gap(a: np.ndarray, b: np.ndarray) -> dict:
    """How far two wire poses ``(3, 4)`` disagree: position gap (m) and rotation gap (rad). Both rotations carry
    the same WODS-style axes, so their relative rotation is proper and its angle is well defined."""
    a, b = np.asarray(a, np.float64).reshape(3, 4), np.asarray(b, np.float64).reshape(3, 4)
    cos = (np.trace(_orthonormal(a[:, :3]).T @ _orthonormal(b[:, :3])) - 1.0) / 2.0
    return {"head_gap_pos": float(np.linalg.norm(a[:, 3] - b[:, 3])),
            "head_gap_rot": float(np.arccos(np.clip(cos, -1.0, 1.0)))}


def _orthonormal(rot: np.ndarray) -> np.ndarray:
    """The nearest orthogonal matrix with ``rot``'s handedness (a policy output need not be exact)."""
    u, _, vt = np.linalg.svd(np.asarray(rot, np.float64))
    out = u @ vt
    if np.sign(np.linalg.det(out)) != np.sign(np.linalg.det(rot)):
        out = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return out


def _base(base_pose) -> tuple[np.ndarray, np.ndarray]:
    """``base_pose`` (xyz + wxyz) -> (rotation (3, 3), position (3,))."""
    import mujoco
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(base_pose[3:7], np.float64))
    return rot.reshape(3, 3), np.asarray(base_pose[:3], np.float64)


def _target_in_base(pose: np.ndarray, base_pose) -> tuple[np.ndarray, np.ndarray]:
    """A wire pose ``(3, 4)`` (house-world <- WODS-style TCP axes) -> the SITE's base-local target (position,
    proper rotation): the axes re-orthonormalised, then z flipped back to the site's GL-style frame."""
    base_rot, base_pos = _base(base_pose)
    site_rot = _orthonormal(pose[:, :3]) @ GL_TO_WODS                   # det -1 -> the site's det +1
    return base_rot.T @ (pose[:, 3] - base_pos), base_rot.T @ site_rot


def _rotation_error(rot: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector (rad, axis * angle) taking ``rot`` to ``target``."""
    import mujoco
    quat, vel = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target @ rot.T).reshape(9))
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel
