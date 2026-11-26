#!/usr/bin/env python3
"""Publish a camera the devbox's ROS 2 stack does not run (the RealSense D455: its colour stream is uncompressed YUYV,
and the stack's camera manager runs MJPEG cameras only) the way the stack publishes its own: JPEG
``sensor_msgs/CompressedImage`` frames on ``--topic``, stamped at capture, with the stack's QoS (reliable, volatile),
so every reader sees it as one more of the devbox's cameras.

It runs on the devbox, in its ROS 2 image (rclpy, OpenCV and numpy from there; nothing from this repo but this file):

  python3 scripts/publish_camera.py --serial <serial> --size 1280 720 --fps 30 --topic /cam_side/image_raw/compressed

``--serial`` finds the camera's colour node itself: a RealSense has several V4L2 nodes (depth, infrared, colour,
metadata) whose numbers change between boots, so it tries each of the camera's nodes and takes the first that gives
frames of ``--size`` in ``--fourcc`` itself (the D455's depth module also serves an imager as UYVY, which OpenCV would
convert and pass off as colour: it shows the IR projector's dots). ``--device`` names the node instead.
"""
import argparse
import glob
import os
import re
import time


def camera_nodes(serial):
    """Every V4L2 node of the USB device with this serial: the nodes its /dev/v4l/by-id links name, and their siblings
    on the same USB device (two of its nodes can share a by-id name, and then only one of them gets the link)."""
    nodes = set()
    for link in glob.glob(f'/dev/v4l/by-id/*{serial}*'):
        video = os.path.basename(os.path.realpath(link))
        usb = os.path.realpath(f'/sys/class/video4linux/{video}/device/..')
        nodes.add(video)
        nodes.update(os.path.basename(p) for p in glob.glob(f'{usb}/*/video4linux/video*'))
    return sorted((f'/dev/{n}' for n in nodes), key=lambda p: int(re.sub(r'\D', '', p) or 0))


def open_capture(device, size, fps, fourcc):
    """An OpenCV V4L2 capture of ``device`` giving ``fourcc`` frames of ``size``, or None."""
    import cv2
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    capture.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = capture.read()
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    got = ''.join(chr((code >> 8 * i) & 0xFF) for i in range(4))      # a RealSense's imager nodes offer UYVY
    if ok and frame is not None and got == fourcc and [frame.shape[1], frame.shape[0]] == list(size):
        return capture
    capture.release()
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument('--serial', help='find the colour node of the USB camera with this serial')
    where.add_argument('--device', help='the V4L2 node to read')
    p.add_argument('--topic', required=True, help='the CompressedImage topic to publish')
    p.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--fourcc', default='YUYV')
    p.add_argument('--jpeg-quality', type=int, default=90)
    p.add_argument('--frame-id', help='the frames\' header frame_id (default: the topic\'s first part)')
    args = p.parse_args()

    import cv2
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    devices = [args.device] if args.device else camera_nodes(args.serial)
    if not devices:
        raise SystemExit(f'no V4L2 node of a camera with serial {args.serial} (is it plugged in, /dev/v4l mounted?)')
    capture, device = None, None
    for device in devices:
        capture = open_capture(device, args.size, args.fps, args.fourcc)
        if capture is not None:
            break
    if capture is None:
        raise SystemExit(f'none of {devices} gives {args.fourcc} {args.size[0]}x{args.size[1]} frames')
    frame_id = args.frame_id or args.topic.strip('/').split('/')[0]
    rclpy.init()
    node = rclpy.create_node(f'publish_{frame_id}')
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    publisher = node.create_publisher(CompressedImage, args.topic, qos)
    quality = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]
    print(f'{device} ({args.fourcc} {args.size[0]}x{args.size[1]} @ {args.fps}) -> {args.topic}', flush=True)
    count, since = 0, time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            stamp = node.get_clock().now().to_msg()                # at capture, as the stack stamps its frames
            if not ok:
                raise SystemExit(f'{device} stopped delivering frames')
            msg = CompressedImage(format='jpeg', data=cv2.imencode('.jpg', frame, quality)[1].tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, frame_id
            publisher.publish(msg)
            count += 1
            if time.monotonic() - since >= 60:
                print(f'{args.topic}: {count / (time.monotonic() - since):.1f} fps', flush=True)
                count, since = 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
