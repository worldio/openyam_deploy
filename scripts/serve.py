#!/usr/bin/env python3
"""The eval server: the real arm behind omniact-molmospaces' Rollout protocol, so OmniAct's eval client
(scripts/eval_omniact.py) drives the robot exactly as it drives omniact-molmospaces' eval_server (README.md, Serve).

  .venv/bin/python scripts/serve.py --config out/openyam_cameras_calibrated_left.json

It serves every calibrated camera's view, holds the vendor's controls from a session's first Reset to its Close, and
never judges success. Each session logs to out/serve/<time>/
and records each episode as a WODS item (out/serve/wods/) and an annotated video. --no-motion runs the same loop with
the arm never commanded; --fake runs it on a simulated robot (no ROS).
"""
import argparse
import json
import signal
import threading
import time
from pathlib import Path

import numpy as np

from openyam_deploy import REPO, arm_settings
from openyam_deploy.fake import FakeRobot, arm_physics
from openyam_deploy.record import Recording
from openyam_deploy.ros import RobotError
from openyam_deploy.server import RolloutService, serve
from openyam_deploy.session import Session, open_options, setup


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='the calibrated camera config of the arm to run (README.md, Calibrate: '
                        'out/openyam_cameras_calibrated_<arm>.json): the cameras, their calibration, the robot topics')
    p.add_argument('--serve-config', default='configs/openyam_serve.json',
                   help='tick, port, and each arm\'s node graph, start pose and gravity compensation; the one of '
                        '--config\'s arm is used')
    p.add_argument('--port', type=int, help='default: the serve config\'s (50051, as omniact-molmospaces\' eval_server)')
    p.add_argument('--no-motion', action='store_true',
                   help='never command the arm: each Step is converted and logged, not sent')
    p.add_argument('--no-gravity-comp', action='store_true', help='command the targets without the gravity offset')
    p.add_argument('--fake', action='store_true', help='a simulated robot instead of ROS (off the robot)')
    p.add_argument('--fake-urdf', default='configs/openyam_robot.urdf', help='--fake: the robot description')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = arm_settings(json.loads((REPO / args.serve_config).read_text()), config)   # the config's arm
    port = args.port or int(settings['port'])
    cams = sorted(name for name, entry in config['cameras'].items() if 'calibration' in entry)
    names = config['motion']['command_joints']
    if args.fake:
        start = np.r_[np.radians(settings['start_pose_deg']), float(settings['reset_gripper_m'])]
        ros = FakeRobot(config, cams, (REPO / args.fake_urdf).read_text(), dict(zip(names, start)))
    else:
        from openyam_deploy.ros import Ros
        ros = Ros(config, cams, command=not args.no_motion, name='openyam_serve')
    try:
        urdf = ros.wait_urdf()
        parts = setup(config, settings, urdf, gravity=not args.no_gravity_comp)
    except RobotError as error:
        ros.close()
        raise SystemExit(str(error)) from None
    log_dir = REPO / settings['log_dir']
    recording = Recording(log_dir / 'wods', config, settings, urdf)
    if args.fake and not args.no_gravity_comp:             # the simulated arm sags back as the real one does
        ros.physics = arm_physics(parts.conv)
    requester = config['motion'].get('owner_requester', 'diagnostics')

    def open_session(bundle, options):
        prompt, views, horizon, continue_on_ik_failure, seed = open_options(bundle, options, parts.views,
                                                                            settings['task_horizon'])
        return Session(ros, parts.emb, parts.conv, settings, prompt=prompt, views=views, task_horizon=horizon,
                       continue_on_ik_failure=continue_on_ik_failure, seed=seed, motion_on=not args.no_motion,
                       requester=requester, log_dir=_session_dir(log_dir), recording=recording)

    service = RolloutService(open_session, settings['idle_timeout_s'])
    server = serve(service, port)
    print(f'serving the Rollout protocol on port {port}: views {parts.views}; gravity compensation '
          f'{"off" if args.no_gravity_comp else "on"}; ' + ('fake robot' if args.fake else
                                                             'arm never commanded' if args.no_motion else 'motion on')
          + f'; logs in {log_dir}, episodes recorded to {recording.wods_root}', flush=True)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        print('shutting down', flush=True)
        service.shutdown()
        server.stop(2.0).wait()
        ros.close()


def _session_dir(root):
    """A new directory under ``root`` named by the time (two sessions in one second get -2, -3, ...)."""
    name = time.strftime('%Y%m%d-%H%M%S')
    for k in range(1, 1000):
        path = root / (name if k == 1 else f'{name}-{k}')
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f'no free session directory for {name} under {root}')


if __name__ == '__main__':
    main()
