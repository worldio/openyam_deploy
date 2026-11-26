"""Run OmniAct policies on the Anvil OpenYAM arm: a Rollout server speaking omniact-molmospaces' eval protocol
(serve.py) and the camera calibration it serves through (calibrate_cameras.py)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]      # the checkout: configs/ and out/ live here


def arm_settings(settings, config):
    """The serve settings (configs/openyam_serve.json: both arms) for the arm of the calibrated camera config
    ``config``, its "arm" (calibrate_cameras.py --arm writes it): the shared settings with the arm's own over them,
    "gravity" key by key. Settings of one arm, as a recording saves them, come back as they are."""
    if 'arms' not in settings:
        return settings
    arm = config.get('arm')
    if arm not in settings['arms']:
        what = 'names no arm' if arm is None else f'is the {arm!r} arm\'s, not one of {sorted(settings["arms"])}'
        raise SystemExit(f'the camera config {what}: give the calibrated config of an arm '
                         '(out/openyam_cameras_calibrated_<arm>.json, from calibrate_cameras.py --arm)')
    spec = settings['arms'][arm]
    out = {k: v for k, v in settings.items() if k not in ('arms', 'gravity')}
    out.update({k: v for k, v in spec.items() if k != 'gravity'})
    out['gravity'] = {**settings.get('gravity', {}), **spec.get('gravity', {})}
    return out
