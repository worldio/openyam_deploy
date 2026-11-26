#!/usr/bin/env python3
"""A live web page of what the policy sees: every calibrated camera's view, reprojected as the server serves it, with
the node graph drawn on it from the arm's live joints, as omniact-molmospaces' visualize_node_graph.py draws a WODS
frame (README.md, Serve). It only listens to the robot (no controls, no commands), so it runs beside the server.

  .venv/bin/python scripts/visualize.py --config out/openyam_cameras_calibrated_left.json

then open http://<robot>:8080 (from elsewhere: ssh -L 8080:localhost:8080 <user>@<robot>). It draws only while a
page is open. --fake draws a simulated robot (no ROS).
"""
import argparse
import json
import signal
import threading
from pathlib import Path

import numpy as np

from openyam_deploy import REPO, arm_settings
from openyam_deploy.embodiment import Embodiment
from openyam_deploy.fake import FakeRobot
from openyam_deploy.live import Live, serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='the calibrated camera config of an arm (out/openyam_cameras_calibrated_<arm>.json): the '
                        'cameras, their calibration and the robot topics')
    p.add_argument('--serve-config', default='configs/openyam_serve.json',
                   help='the node graph of each arm; the one of --config\'s arm is drawn')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--host', default='0.0.0.0', help='the address to listen on (127.0.0.1: this machine only)')
    p.add_argument('--fps', type=float, default=10.0, help='the most drawings per second')
    p.add_argument('--scale', type=int, default=2, help='each view drawn at this many times its size')
    p.add_argument('--fake', action='store_true', help='a simulated robot instead of ROS (off the robot)')
    p.add_argument('--fake-urdf', default='configs/openyam_robot.urdf', help='--fake: the robot description')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = arm_settings(json.loads((REPO / args.serve_config).read_text()), config)   # the config's arm
    cameras = {name: entry for name, entry in config['cameras'].items() if 'calibration' in entry}
    if not cameras:
        raise SystemExit('no calibrated camera in the config: run the calibration first (README.md, Calibrate)')
    if args.fake:
        names = config['motion']['command_joints']
        start = np.r_[np.radians(settings['start_pose_deg']), float(settings['reset_gripper_m'])]
        ros = FakeRobot(config, sorted(cameras), (REPO / args.fake_urdf).read_text(), dict(zip(names, start)))
    else:
        from openyam_deploy.ros import Ros
        ros = Ros(config, sorted(cameras), command=False, name='openyam_visualize')
    try:
        emb = Embodiment(settings['node_graph'], cameras, ros.wait_urdf(), config['reference_body'])
        live = Live(ros, emb, cameras, args.fps, args.scale)
        server = serve(live, args.port, args.host)
    except BaseException:
        ros.close()
        raise
    print(f'drawing {sorted(cameras)} on http://{args.host}:{args.port} (Ctrl-C stops)', flush=True)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        server.shutdown()
        live.close()
        ros.close()


if __name__ == '__main__':
    main()
