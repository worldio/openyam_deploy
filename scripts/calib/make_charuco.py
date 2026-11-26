#!/usr/bin/env python3
"""Print the ChArUco board a camera declares in the calibration config, as a US-Letter PNG.

The board spec lives ONLY in the config (cameras.<name>.board: dict / squares_x / squares_y /
square_len_mm / marker_ratio) -- the single source of truth that calibrate_cameras.py also reads,
so there is no board.json to keep in sync. This renders the exact matching board centered on a
US-Letter sheet. Print at 100% (no fit-to-page); then caliper a square and, only if the printer
rescaled it, set the measured square_len_mm back in the config.

  python3 scripts/calib/make_charuco.py --config configs/openyam_left_cameras.json \
      --camera wrist_l --out-dir out/board_wrist_l
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='calibration config with cameras.<name>.board')
    p.add_argument('--camera', required=True, help='which camera\'s board to print')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--margin-mm', type=float, default=10.0, help='min white quiet zone around the board')
    p.add_argument('--orientation', choices=['auto', 'portrait', 'landscape'], default='auto')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    cam = config.get('cameras', {}).get(args.camera)
    if cam is None or 'board' not in cam:
        raise SystemExit(f'{args.camera}: no "board" in config; cameras: {list(config.get("cameras", {}))}')
    spec = cam['board']
    sx, sy = int(spec['squares_x']), int(spec['squares_y'])
    sq = float(spec['square_len_mm'])
    ratio = float(spec.get('marker_ratio', 0.75))
    square_m, marker_m = sq / 1000.0, sq / 1000.0 * ratio
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dict']))
    board = cv2.aruco.CharucoBoard((sx, sy), square_m, marker_m, dictionary)

    mm2px = lambda mm: int(round(mm / 25.4 * args.dpi))
    w, h = mm2px(sx * sq), mm2px(sy * sq)
    board_in = (sx * sq / 25.4, sy * sq / 25.4)
    pad_in = args.margin_mm / 25.4

    def fits(pw, ph):
        return board_in[0] + 2 * pad_in <= pw and board_in[1] + 2 * pad_in <= ph
    if args.orientation == 'portrait':
        pw, ph = 8.5, 11.0
    elif args.orientation == 'landscape':
        pw, ph = 11.0, 8.5
    elif fits(8.5, 11.0):
        pw, ph = 8.5, 11.0
    elif fits(11.0, 8.5):
        pw, ph = 11.0, 8.5
    else:
        raise SystemExit(f'board {board_in[0]:.1f}x{board_in[1]:.1f}in + margin does not fit US Letter; '
                         f'reduce squares/square_len_mm in the config')
    W, H = int(round(pw * args.dpi)), int(round(ph * args.dpi))
    sheet = np.full((H, W), 255, np.uint8)
    img = board.generateImage((w, h), marginSize=0, borderBits=1)
    ox, oy = (W - w) // 2, (H - h) // 2
    sheet[oy:oy + h, ox:ox + w] = img
    caption = (f'{args.camera}: {spec["dict"]} {sx}x{sy}  square={sq:g}mm  marker={sq * ratio:.2f}mm  '
               f'-- print 100% (US Letter), then MEASURE a square')
    cv2.putText(sheet, caption, (ox, min(H - mm2px(4), oy + h + mm2px(6))),
                cv2.FONT_HERSHEY_SIMPLEX, args.dpi / 900.0, 0, max(1, args.dpi // 300), cv2.LINE_AA)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f'charuco_{args.camera}_{sx}x{sy}_{sq:g}mm.png'
    cv2.imwrite(str(png), sheet)
    cc, ci, mc, mi = cv2.aruco.CharucoDetector(board).detectBoard(sheet)
    n_expected, n_found = (sx - 1) * (sy - 1), (0 if ci is None else len(ci))
    print(f'wrote {png}  (US Letter {pw:g}x{ph:g}in = {W}x{H}px @ {args.dpi}dpi; '
          f'board {sx * sq:g}x{sy * sq:g}mm centered)')
    print(f'self-test: detected {n_found}/{n_expected} inner corners, '
          f'{0 if mi is None else len(mi)} markers  ({"OK" if n_found == n_expected else "CHECK"})')


if __name__ == '__main__':
    main()
