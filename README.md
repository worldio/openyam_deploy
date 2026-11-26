# openyam_deploy

Run [OmniAct](https://github.com/worldio/omniact) policies on the Anvil OpenYAM arm. `scripts/serve.py` puts the real arm behind the Rollout gRPC protocol of the [omniact-molmospaces](https://github.com/worldio/omniact-molmospaces) evaluation server, so OmniAct's eval client drives the robot as it drives the simulator, and the client decides the task.

<p align="center">
  <a href="https://arxiv.org/abs/2611.33158"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="https://omniact.github.io/"><b>Project page</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/worldio/omniact"><b>Code</b></a> &nbsp;·&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

## Setup

On the robot's computer, which runs the vendor's ROS 2 stack:

```bash
git clone https://github.com/worldio/openyam_deploy.git && cd openyam_deploy
```

Every command below runs in a shell of the vendor's ROS 2 image. Set `<interface>` and `<peer-ip>`, and `ROS_DOMAIN_ID` if it differs, from the vendor's own DDS settings:

```bash
docker run --rm -it --network host --user "$(id -u):$(id -g)" \
  --volume "$PWD:$PWD" --workdir "$PWD" --volume "$(command -v uv):/usr/local/bin/uv:ro" \
  --env HOME=/tmp --env PYTHONUNBUFFERED=1 --env UV_PYTHON_PREFERENCE=only-system \
  --env ROS_DOMAIN_ID=199 --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --env CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="<interface>"/></Interfaces></General><Discovery><Peers><Peer address="<peer-ip>"/></Peers><ParticipantIndex>auto</ParticipantIndex></Discovery><Internal><SocketReceiveBufferSize min="128MB"/></Internal></Domain></CycloneDDS>' \
  --entrypoint bash "$(docker inspect --format '{{.Image}}' anvil-loader-ros2-1)" \
  -c 'source /workspace/install/setup.bash && exec bash'
```

In that shell, the first time:

```bash
uv sync --frozen
```

## Calibrate

The calibration writes `out/openyam_cameras_calibrated_left.json`, which the first command starts from `configs/openyam_cameras.json` and every later command extends.

Fill in each camera's `raw` lens in `configs/openyam_cameras.json` from its factory intrinsics (ours are there now):

- `lens`: `fisheye` or `pinhole`;
- `size`: the `[width, height]` it streams at;
- `K`: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, in pixels at that size;
- `D`: OpenCV's distortion coefficients, `[k1, k2, k3, k4]` for `fisheye` or `[k1, k2, p1, p2, k3]` for `pinhole`.

1. Print both boards at 100%. If a printed square measures differently, set `square_len_mm` in `configs/openyam_cameras.json`. Mount the gripper board on the gripper and fix the fixed board flat on the table below it.
   ```bash
   .venv/bin/python scripts/calib/make_charuco.py --config configs/openyam_cameras.json --camera waist --out-dir out/boards     # the gripper board (waist, side)
   .venv/bin/python scripts/calib/make_charuco.py --config configs/openyam_cameras.json --camera wrist_l --out-dir out/boards   # the fixed board (wrist_l)
   ```
2. Publish the side camera if the vendor stack doesn't run it. Ours is a RealSense, but `scripts/publish_camera.py` works with any V4L2 camera: `--serial` finds it (or `--device` names its node), and `--size`, `--fps` and `--fourcc` set its stream. The publisher runs in its own container until `docker rm -f cam-side`:
   ```bash
   docker run -d --name cam-side --network host --user "$(id -u):$(id -g)" --group-add "$(getent group video | cut -d: -f3)" \
     $(for d in /dev/video*; do printf -- '--device %s ' "$d"; done) --volume /dev/v4l:/dev/v4l:ro \
     --volume "$PWD:$PWD:ro" --workdir "$PWD" --env HOME=/tmp --env PYTHONUNBUFFERED=1 \
     --env ROS_DOMAIN_ID=199 --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
     --env CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="<interface>"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex></Discovery><Internal><SocketReceiveBufferSize min="128MB"/></Internal></Domain></CycloneDDS>' \
     --entrypoint bash "$(docker inspect --format '{{.Image}}' anvil-loader-ros2-1)" \
     -c 'source /workspace/install/setup.bash && exec python3 scripts/publish_camera.py --serial <serial> --size 1280 800 --fps 30 --topic /cam_side/image_raw/compressed'
   ```
3. Calibrate the views. Each jog moves the arm, so start each one where the camera's view shows its board. Run each calibration until about 200 valid captures (the `board frames` count it prints); Ctrl-C ends it. For the exo cameras' calibration it will try to aim the gripper at preset locations. If they're different in your setup, update each camera's `view.R` in `configs/openyam_cameras.json` first.
   ```bash
   # wrist_l: the fixed board on the table below the gripper
   .venv/bin/python scripts/calibrate_cameras.py --arm left --jog --cameras wrist_l \
       --config out/openyam_cameras_calibrated_left.json --config-out out/openyam_cameras_calibrated_left.json --work-dir out/calib-left-wrist_l
   # waist: the gripper in front of the base, its board facing the waist camera
   .venv/bin/python scripts/calibrate_cameras.py --arm left --jog --cameras waist --present-waist \
       --config out/openyam_cameras_calibrated_left.json --config-out out/openyam_cameras_calibrated_left.json --work-dir out/calib-left-waist
   # side: the gripper board turned toward the side camera
   .venv/bin/python scripts/calibrate_cameras.py --arm left --jog --cameras side \
       --config out/openyam_cameras_calibrated_left.json --config-out out/openyam_cameras_calibrated_left.json --work-dir out/calib-left-side
   ```
4. Check the result on the live page at `http://<robot-ip>:8080`: the node graph drawn on every view should lie on the arm. A solve that printed a WARNING stopped early, so jog again. To re-solve a capture without moving, rerun its command with `--reuse-capture` in place of `--jog`.
   ```bash
   .venv/bin/python scripts/visualize.py --config out/openyam_cameras_calibrated_left.json
   ```

## Serve

```bash
.venv/bin/python scripts/serve.py --config out/openyam_cameras_calibrated_left.json
```

`--no-motion` runs everything but never commands the arm, and `--no-gravity-comp` commands the targets without the gripper's gravity offset. Episodes are recorded under `out/serve/`. In a second shell, watch what the policy sees at `http://<robot-ip>:8080`:

```bash
.venv/bin/python scripts/visualize.py --config out/openyam_cameras_calibrated_left.json
```

On the GPU machine, OmniAct's `scripts/eval_omniact.py` connects with `--target <robot-ip>:50051` and picks the task with `--prompt`.

## Citation

```bibtex
@misc{fan2026omniactpolicymillionembodiments,
      title={OmniAct: One Policy, One Million Embodiments},
      author={Xiang Fan and Yuheng Wang and Jiayuan Yan and Bohan Fang and Wisdom Oluchi Ikezogwo and Ainaz Eftekhar and Max Argus and Zhongzheng Ren and Jiafei Duan and Ranjay Krishna},
      year={2026},
      eprint={2611.33158},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.33158},
}
```
