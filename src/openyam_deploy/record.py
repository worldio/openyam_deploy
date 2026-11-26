"""Every served episode, recorded: a WODS item with the fields and conventions of omniact-molmospaces' datagen items
(its scripts/embodiments/embodiment_wods.py), and a video of the episode with what the server did to each step drawn
on it.

The session collects an episode as it runs (``Episode``: the observations the client got, the measured joints, each
step's record) and hands it over when it ends (done, a new Reset, Close, or an error that stopped a step). Its log
thread saves it as ``<session>/ep<k>.npz`` + ``ep<k>.json`` and starts a separate, niced process

  python -m openyam_deploy.record <session>/ep<k>.npz

which writes the WODS item (``<wods root>/<session>_ep<k>/openyam``) and ``<session>/ep<k>.mp4``, then deletes the
npz. Encoding never runs in the server's process, so it never competes with the control loop. If the process fails,
the npz stays and the same command redoes it.

The WODS item, one shard per view (the views as served, 256 px), T frames; frame i is observation i (0: the Reset's):
  video, intrinsics (the view's K), c2w (WODS camera axes -> the arm base), node_geometry, node_pose, node_proprio,
  node_graph: the observation, as the client got it
  proprio      (T, 7)     the measured command joints (arm rad, then the gripper's per-finger m)
  action       (T, N)     the step taken from frame i as datagen's label: per node, the target the server commanded
                          (after the IK and the joint limits; before the gravity offset) minus the
                          node proprio. The last frame holds the last target.
  node_action_geometry, node_action_pose: the nodes and the TCP at that target (the pose in the arm base)
  timestamps   i * tick_s, as datagen's policy steps (the real times are in steps.jsonl and on the video)
  caption      the prompt

The video shows, per frame, the node graph on every view (top: the observation, bottom: the step's target) with the
TCP the policy asked for (magenta), where the server sent it (the target's axes) and where the arm got by the next
frame (cyan), and beside them the numbers: the IK's outcome, clipped joints, the gravity offset, the gripper and the
timing.
"""
import functools
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

EMBODIMENT = 'openyam'           # the key's last part: an item per embodiment, as OmniAct's keys (<episode>/<emb>)
OBS_KEYS = ('views', 'node_geometry', 'node_proprio', 'node_pose', 'c2w')
SESSION_FILES = ('cameras.json', 'serve.json', 'robot.urdf')    # what the recorder rebuilds the kinematics from
REQUEST, REACHED = (255, 70, 255), (60, 230, 255)               # the policy's TCP, the arm's TCP at the next frame
TEXT_PX = 8.5                                                   # the video's text size, at scale 1


class Recording:
    """Where a session's episodes go: the WODS root, and what the recorder needs (the calibrated camera config, the
    serve settings and the URDF, saved beside the session's logs). ``spawn`` False leaves the writing to the caller."""

    def __init__(self, wods_root, config, settings, urdf_text, spawn=True):
        self.wods_root, self.spawn = Path(wods_root), spawn
        self.files = dict(zip(SESSION_FILES, (json.dumps(config, indent=1), json.dumps(settings, indent=1),
                                              urdf_text)))

    def begin(self, directory):
        for name, text in self.files.items():
            (directory / name).write_text(text)

    def finish(self, episode, directory):
        """Save the episode, then (with ``spawn``) start the process that writes it; returns the npz path."""
        path = save(episode, directory, self.wods_root)
        if path is not None and self.spawn:
            with open(path.with_suffix('.log'), 'w') as log:
                process = subprocess.Popen([sys.executable, '-m', 'openyam_deploy.record', str(path)], stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)   # outlives a Ctrl-C
            os.setpriority(os.PRIO_PROCESS, process.pid, 10)
        return path


class Episode:
    """One episode as the session ran it: each observation (frame i) with the measured command joints and its time
    since the first, and each step's record (its steps.jsonl line: the request, what the server made of it)."""

    def __init__(self, index, prompt, command_joints, tick_s, motion_on):
        self.index, self.prompt, self.names = index, prompt, list(command_joints)
        self.tick_s, self.motion_on = float(tick_s), bool(motion_on)
        self.obs, self.joints, self.times, self.steps, self.end = [], [], [], [], None
        self.started = time.time()
        self._t0 = None

    def frame(self, obs, joints):
        now = time.monotonic()
        self._t0 = now if self._t0 is None else self._t0
        self.obs.append(obs)
        self.joints.append([float(joints[name]) for name in self.names])
        self.times.append(now - self._t0)

    def step(self, record):
        self.steps.append(record)


def save(episode, directory, wods_root):
    """``<directory>/ep<k>.npz`` (the observations, joints and times) and ``ep<k>.json`` (the rest); None for an
    episode without an observation."""
    if not episode.obs:
        return None
    path = Path(directory) / f'ep{episode.index:02d}.npz'
    np.savez(path, joints=np.asarray(episode.joints), times=np.asarray(episode.times),
             **{key: np.stack([np.asarray(obs[key]) for obs in episode.obs]) for key in OBS_KEYS})
    first = episode.obs[0]
    meta = {'index': episode.index, 'key': f'{Path(directory).name}_ep{episode.index:02d}/{EMBODIMENT}',
            'wods_root': str(wods_root), 'prompt': episode.prompt, 'end': episode.end,
            'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(episode.started)),
            'tick_s': episode.tick_s, 'motion': episode.motion_on, 'command_joints': episode.names,
            'camera_names': list(first['camera_names']), 'node_names': list(first['node_names']),
            'edges': np.asarray(first['edges']).tolist(), 'axis_mask': np.asarray(first['axis_mask']).tolist(),
            'joint_type': np.asarray(first['joint_type']).tolist(), 'steps': episode.steps}
    path.with_suffix('.json').write_text(json.dumps(meta, default=_plain))
    return path


# ---- the recorder process
def write(path, video=True, scale=2):
    """Write the episode saved at ``path`` (ep<k>.npz): the WODS item, then (with ``video``) the annotated video;
    then delete the npz. Returns the WODS key."""
    from .embodiment import Embodiment

    path = Path(path)
    directory, meta = path.parent, json.loads(path.with_suffix('.json').read_text())
    config = json.loads((directory / 'cameras.json').read_text())
    settings = json.loads((directory / 'serve.json').read_text())
    cameras = {name: entry for name, entry in config['cameras'].items() if 'calibration' in entry}
    emb = Embodiment(settings['node_graph'], cameras, (directory / 'robot.urdf').read_text(), config['reference_body'])
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    episode = _derive(emb, meta, arrays)
    intrinsics = {name: emb.views[name].view['K'] for name in meta['camera_names']}
    _write_wods(meta, arrays, episode, intrinsics)
    if video:
        _write_video(path.with_suffix('.mp4'), meta, arrays, episode, intrinsics, scale)
    path.unlink()
    return meta['key']


def _derive(emb, meta, arrays):
    """Per frame: the target the step from it commanded (the last frame, and a step that never converted, hold the
    one before; frame 0 without one holds the measured joints), the relative node action to it, the nodes and TCP at
    it (the nodes in each view's camera frame at that frame), and the TCP the policy asked for, if any."""
    names, n_frames = meta['command_joints'], len(arrays['times'])
    steps = {record['step']: record for record in meta['steps']}
    targets, target = [], arrays['joints'][0]
    for i in range(n_frames):
        if 'target' in steps.get(i, {}):
            target = np.asarray(steps[i]['target'], float)
        targets.append(target)
    n_nodes, n_views = arrays['node_proprio'].shape[1], arrays['c2w'].shape[1]
    action = np.zeros((n_frames, n_nodes), np.float32)
    action_geometry = np.zeros((n_frames, n_nodes, 2, n_views, 3), np.float32)
    action_pose = np.zeros((n_frames, n_nodes, 3, 4), np.float32)
    request = np.full((n_frames, 3, 4), np.nan)
    joint = emb.axis_mask
    for i, target in enumerate(targets):
        points, proprio, tcp = emb.geometry(dict(zip(names, target)))
        action[i, joint] = proprio[joint] - arrays['node_proprio'][i, joint]
        for v in range(n_views):
            c2w = arrays['c2w'][i, v]
            action_geometry[i, :, :, v] = (points - c2w[:, 3]) @ c2w[:, :3]
        action_pose[i, emb.tcp_node] = tcp[:3]
        record = steps.get(i, {})
        if record.get('pose_mask') and record['pose_mask'][emb.tcp_node]:
            request[i] = record['tcp_target']
    return {'targets': np.asarray(targets), 'action': action, 'action_geometry': action_geometry,
            'action_pose': action_pose, 'request': request, 'steps': steps, 'tcp_node': emb.tcp_node}


def _write_wods(meta, arrays, episode, intrinsics):
    import torch
    from wods.v2 import write_wods_key_done, write_wods_metadata, write_wods_shard, write_wods_shard_done

    root, key, n_frames = meta['wods_root'], meta['key'], len(arrays['times'])

    def tensor(value, dtype=torch.float32):
        return torch.from_numpy(np.ascontiguousarray(value)).to(dtype)

    types = torch.tensor(meta['joint_type'], dtype=torch.long)
    graph = {'edges': torch.tensor(meta['edges'], dtype=torch.long).reshape(-1, 2),
             'axis_mask': (types == 1) | (types == 2), 'joint_type': types, 'node_names': list(meta['node_names'])}
    timestamps = torch.arange(n_frames, dtype=torch.float32) * meta['tick_s']
    write_wods_metadata(root, key, caption=meta['prompt'])
    for v, camera in enumerate(meta['camera_names']):
        write_wods_shard(
            root, key, camera, video=tensor(arrays['views'][:, v], torch.uint8), timestamps=timestamps,
            intrinsics=tensor(np.repeat(intrinsics[camera][None], n_frames, axis=0)), c2w=tensor(arrays['c2w'][:, v]),
            proprio=tensor(arrays['joints']), action=tensor(episode['action']),
            node_proprio=tensor(arrays['node_proprio']), node_graph=graph,
            node_geometry=tensor(arrays['node_geometry'][:, :, :, v]),
            node_action_geometry=tensor(episode['action_geometry'][:, :, :, v]),
            node_pose=tensor(arrays['node_pose'][:, :, v]), node_action_pose=tensor(episode['action_pose']))
        write_wods_shard_done(root, key, camera)
    write_wods_key_done(root, key)            # LAST: the marker is what makes the item readable


# ---- the annotated video
def _write_video(path, meta, arrays, episode, intrinsics, scale):
    import av

    frames = len(arrays['times'])
    container = av.open(str(path), 'w')
    stream = container.add_stream('libx264', rate=Fraction(1.0 / meta['tick_s']).limit_denominator(1000),
                                  options={'crf': '23'})
    stream.pix_fmt = 'yuv420p'
    try:
        for i in range(frames):
            image = _video_frame(i, meta, arrays, episode, intrinsics, scale)
            if i == 0:
                stream.width, stream.height = image.shape[1], image.shape[0]
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _video_frame(i, meta, arrays, episode, intrinsics, scale, margin=(0.1, 0.2)):
    """Frame i: [numbers | views with the observation's graph] over [timeline | views with the step's target]."""
    from PIL import Image

    from .nodegraph import node_types, pose_points, project_cam_frame, projected, render_view

    names, types = meta['node_names'], node_types(meta['joint_type'])
    edges, axis_mask = [tuple(e) for e in meta['edges']], np.asarray(meta['axis_mask'], bool)
    last, tcp = len(arrays['times']) - 1, episode['tcp_node']
    reached = None
    if i < last:                                       # the TCP at the next frame, in the arm base
        reached = (_se3(arrays['c2w'][i + 1, 0]) @ _se3(arrays['node_pose'][i + 1, tcp, 0]))[:3]
    top, bottom = [], []
    for v, camera in enumerate(meta['camera_names']):
        frame, K = arrays['views'][i, v], intrinsics[camera]
        H, W = frame.shape[:2]
        c2w = arrays['c2w'][i, v]

        def poses(track):
            return [tuple(map(tuple, p)) for p in project_cam_frame(pose_points(track[None])[0], K, (H, W))]

        top.append(render_view(frame, projected(arrays['node_geometry'][i, :, :, v], K, (H, W), axis_mask), edges,
                               types, names, W, H, scale, margin, title=camera,
                               poses=poses(arrays['node_pose'][i, :, v])))
        target_pose = pose_points(episode['action_pose'][i][None], c2w[None])[0]
        panel = render_view(frame, projected(episode['action_geometry'][i, :, :, v], K, (H, W), axis_mask), edges,
                            types, names, W, H, scale, margin, title=f'{camera}: target',
                            poses=[tuple(map(tuple, p)) for p in project_cam_frame(target_pose, K, (H, W))])
        marks = [(pose, colour) for pose, colour in ((episode['request'][i], REQUEST), (reached, REACHED))
                 if pose is not None and np.isfinite(pose).all()]
        bottom.append(_mark(panel, marks, c2w, K, (H, W), scale, margin))
    height = min(p.shape[0] for p in top + bottom)
    width = top[0].shape[1]
    left_top = _numbers(i, meta, arrays, episode, reached, (width, height), scale)
    left_bottom = _timeline(i, meta, arrays, episode, (width, height), scale)
    image = np.concatenate([np.concatenate([left_top] + [p[:height] for p in top], axis=1),
                            np.concatenate([left_bottom] + [p[:height] for p in bottom], axis=1)], axis=0)
    image = np.ascontiguousarray(image[:image.shape[0] // 2 * 2, :image.shape[1] // 2 * 2])
    return np.asarray(Image.fromarray(image).convert('RGB'))


def _mark(panel, marks, c2w, K, size, scale, margin):
    """Draw each (TCP pose 3x4 in the arm base, colour) on a render_view panel: a ring at the TCP and its approach."""
    from PIL import Image, ImageDraw

    from .nodegraph import POSE_LEN, project_cam_frame

    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    (H, W), (mx, my) = size, margin
    for pose, colour in marks:
        world = np.array([pose[:, 3], pose[:, 3] + POSE_LEN * pose[:, 2]])
        x, y, z = project_cam_frame((world - c2w[:, 3]) @ c2w[:, :3], K, (H, W)).T
        if z.min() <= 0:
            continue
        u, w = (mx + x) * W * scale, (my + y) * H * scale
        r, line = 5.0 * scale, max(2, scale)
        for fill, width in (((0, 0, 0), 3 * line), (colour, line)):     # a dark rim, then the colour
            draw.line([(u[0], w[0]), (u[1], w[1])], fill=fill, width=width)
            draw.ellipse([u[0] - r, w[0] - r, u[0] + r, w[0] + r], outline=fill, width=width)
    return np.asarray(image)


def _numbers(i, meta, arrays, episode, reached, size, scale):
    """The frame's text: the step taken from it and what the server did to it."""
    record, last = episode['steps'].get(i), len(arrays['times']) - 1
    ok, warn, bad, dim = (225, 225, 230), (255, 215, 60), (255, 90, 80), (140, 140, 150)
    lines = [(line, ok) for line in textwrap.wrap(meta['prompt'] or '(no prompt)', _columns(size, scale))[:3]]
    lines += [('', ok), (f'frame {i} / {last}   t {arrays["times"][i]:6.2f} s', ok)]
    if record is None:
        lines += [('', ok), (f'end: {meta["end"] or "-"}', warn if i == last else dim)]
        return _text_panel(lines, size, scale)
    timing = record.get('timing') or {}
    ms = {k: 1000 * timing[k] for k in ('client_s', 'server_s', 'convert_s', 'move_s', 'observe_s')
          if timing.get(k) is not None}
    lines.append(('policy ' + (f'{ms["client_s"]:.0f} ms' if 'client_s' in ms else '-') + '   server '
                  + (f'{ms["server_s"]:.0f} ms' if 'server_s' in ms else '-'), dim))
    pose_on = bool(record.get('pose_mask')) and record['pose_mask'][episode['tcp_node']]
    masked = [name for name, on in zip(meta['node_names'], record.get('action_mask') or ()) if on]
    arm = [name for name in masked if not name.startswith('gripper/')]
    heads = ((['TCP pose'] if pose_on else []) + ([f'{len(arm)} arm joint' + 's' * (len(arm) > 1)] if arm else [])
             + (['gripper'] if len(arm) < len(masked) else []))
    lines += [('', ok), ('asked   ' + (' + '.join(heads) or 'nothing'), ok)]
    if 'ik_outcome' in record:
        failed = record.get('ik_failed')
        lines.append((f'IK      {record["ik_outcome"]}   {_mm(record.get("ik_pos_err"))} mm'
                      f'  {_deg(record.get("ik_rot_err"))} deg', bad if failed else ok))
    clipped = record.get('clipped')
    lines.append(('limits  ' + (', '.join(c.replace('follower_l_', '') for c in clipped) if clipped else '-'),
                  warn if clipped else ok))
    gravity = record.get('gravity_offset_deg')
    lines.append(('gravity ' + (' '.join(f'{g:+.2f}' for g in gravity) + ' deg' if gravity else 'off'), ok))
    target = episode['targets'][i]
    lines.append((f'gripper {1000 * arrays["joints"][i][-1]:.1f} -> {1000 * target[-1]:.1f} mm', ok))
    commanded = episode['action_pose'][i]
    if pose_on and np.isfinite(episode['request'][i]).all():
        lines.append((f'target - asked    {_gap(commanded[episode["tcp_node"]], episode["request"][i])}', ok))
    if reached is not None:
        lines.append((f'reached - target  {_gap(reached, commanded[episode["tcp_node"]])}', ok))
    if record.get('error'):
        lines += [('', ok)] + [(line, bad) for line in textwrap.wrap(record['error'], _columns(size, scale))[:3]]
    if i == last:
        lines += [('', ok), (f'end: {meta["end"] or "-"}', warn)]
    return _text_panel(lines, size, scale)


def _timeline(i, meta, arrays, episode, size, scale):
    """The legend, and per step what the server changed (rows)."""
    from PIL import Image, ImageDraw

    from .nodegraph import get_font

    width, height = size
    u = TEXT_PX * scale                                        # the text size in pixels: the layout's unit
    image = Image.new('RGB', size, (26, 26, 30))
    draw, font, label = ImageDraw.Draw(image), get_font(int(u)), (190, 190, 200)
    y = 0.5 * u
    for text, colour in (('asked by the policy', REQUEST), ('reached by the next frame', REACHED),
                         ('the target: its graph and axes', (225, 225, 230))):
        draw.ellipse([0.7 * u, y + 0.15 * u, 1.5 * u, y + 0.95 * u], outline=colour, width=max(2, scale))
        draw.text((2 * u, y), text, fill=colour, font=font)
        y += 1.4 * u
    frames, steps = len(arrays['times']), episode['steps']
    rows = (('IK failed', lambda r: r.get('ik_failed'), (255, 90, 80)),
            ('clipped', lambda r: r.get('clipped'), (200, 120, 255)))
    left, right = 7.5 * u, width - 0.7 * u
    column = (right - left) / max(frames, 1)
    y += 0.8 * u
    rows_top = y
    for text, hit, colour in rows:
        draw.text((0.7 * u, y), text, fill=label, font=font)
        draw.rectangle([left, y + 0.1 * u, right, y + 0.9 * u], outline=(70, 70, 78))
        for k, record in steps.items():
            if k < frames and hit(record):
                draw.rectangle([left + k * column, y + 0.1 * u, left + (k + 1) * column, y + 0.9 * u], fill=colour)
        y += 1.3 * u
    x = left + (i + 0.5) * column
    draw.line([(x, rows_top), (x, y - 0.3 * u)], fill=(235, 235, 240), width=1)
    return np.asarray(image)


def _text_panel(lines, size, scale):
    from PIL import Image, ImageDraw

    image = Image.new('RGB', size, (26, 26, 30))
    draw, font = ImageDraw.Draw(image), _mono_font(int(TEXT_PX * scale))
    for k, (text, colour) in enumerate(lines):
        draw.text((6 * scale, 4 * scale + k * 1.4 * TEXT_PX * scale), text, fill=colour, font=font)
    return np.asarray(image)


def _columns(size, scale):
    return int((size[0] - 12 * scale) / (0.61 * TEXT_PX * scale))      # DejaVu Sans Mono: 0.6 em per character


@functools.lru_cache(maxsize=4)
def _mono_font(size):
    from matplotlib import font_manager
    from PIL import ImageFont

    try:
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans Mono'), size)
    except OSError:
        return ImageFont.load_default()


def _se3(pose):
    out = np.eye(4)
    out[:3] = pose
    return out


def _gap(a, b):
    """'<mm> mm <deg> deg' between two 3x4 poses."""
    angle = math.degrees(math.acos(float(np.clip((np.trace(a[:, :3].T @ b[:, :3]) - 1) / 2, -1, 1))))
    return f'{1000 * np.linalg.norm(a[:, 3] - b[:, 3]):5.1f} mm {angle:4.1f} deg'


def _mm(value):
    return '-' if value is None else f'{1000 * float(value):.1f}'


def _deg(value):
    return '-' if value is None else f'{math.degrees(float(value)):.1f}'


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == '__main__':
    began = time.monotonic()
    written = write(sys.argv[1])
    print(f'{written}: written in {time.monotonic() - began:.1f} s', flush=True)
