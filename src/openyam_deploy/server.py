"""The Rollout gRPC service (proto/rollout.proto): the same four RPCs as omniact-molmospaces' eval_server, over one
Session at a time (one robot).

Status codes follow omniact-molmospaces' where the client depends on them: a malformed request or action is
INVALID_ARGUMENT, an unknown session NOT_FOUND; FAILED_PRECONDITION, which OmniAct's eval client takes as "skip this
key", is never sent. A second Open while a session is live is RESOURCE_EXHAUSTED; motion that had to stop (stale
joints, the hardware paused or e-stopped, the controls taken, the client gone) is ABORTED; missing sensor data or a
vendor refusal is UNAVAILABLE; anything else is INTERNAL with its message. A client that goes away mid-Step stops the
arm, and a session idle for ``idle_timeout`` s is closed (its controls given back).
"""
import threading
import time
import traceback
import uuid
from concurrent import futures
from contextlib import contextmanager

import grpc

from . import codec
from .motion import Stopped
from .proto import rollout_pb2 as pb
from .proto import rollout_pb2_grpc as rpc
from .ros import NotReady, RobotError

MAX_MESSAGE = 512 * 1024 * 1024        # as omniact-molmospaces': the client sends the whole replay bundle in Open
STEP_FIELDS = ('action', 'node_action_pose', 'action_mask', 'node_action_pose_mask')


class Busy(RuntimeError):
    """Another session holds the robot."""


class UnknownSession(KeyError):
    """No such session."""


class RolloutService(rpc.RolloutServicer):
    """``open_session(bundle, options)`` makes a Session for an Open."""

    def __init__(self, open_session, idle_timeout):
        self.open_session, self.idle_timeout = open_session, float(idle_timeout)
        self.session, self.session_id, self.last_call = None, None, 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def Open(self, request, context):
        with self.lock, _status(context):
            if self.session is not None and self.session.stopped:   # its client went away
                self._close('its client went away mid-step')
            if self.session is not None:
                raise Busy('another session holds the robot (one robot, one session); it ends with its client\'s '
                           f'Close, or after {self.idle_timeout:.0f} s idle')
            self.session = self.open_session(dict(request.bundle), dict(request.options))
            self.session_id, self.last_call = uuid.uuid4().hex, time.monotonic()
            print(f'session {self.session_id[:8]} opened: {self.session.prompt!r}', flush=True)
            return pb.Session(session_id=self.session_id, n_episodes=self.session.n_episodes)

    def Reset(self, request, context):
        with self.lock, _status(context):
            return codec.observation(self._get(request.session_id).reset(int(request.episode_idx)))

    def Step(self, request, context):
        with self.lock, _status(context):
            session = self._get(request.session_id)
            missing = [f for f in STEP_FIELDS if not request.HasField(f)]
            if missing:
                raise ValueError(f'Step needs all four action tensors; missing {missing}')
            finished = threading.Event()
            context.add_callback(lambda: finished.is_set() or session.stop())   # the client gone: stop the arm
            try:
                obs, done, info = session.step(*(codec.decode(getattr(request, f)) for f in STEP_FIELDS))
            finally:
                finished.set()
            self.last_call = time.monotonic()
            return pb.StepResult(obs=codec.observation(obs), done=done, info=codec.info(info))

    def Close(self, request, context):
        if self.session is not None and request.session_id == self.session_id:
            self.session.stop()                                   # interrupt a running Step before waiting on it
        with self.lock:
            if request.session_id == self.session_id:
                print(f'session {self.session_id[:8]} closed', flush=True)
                self._close('the client closed')
        return pb.Empty()

    def shutdown(self):
        if self.session is not None:
            self.session.stop()
        with self.lock:
            self._close('the server shut down')

    def _get(self, session_id):
        if self.session is None or session_id != self.session_id:
            raise UnknownSession('unknown or expired session')
        self.last_call = time.monotonic()
        return self.session

    def _close(self, reason):
        session, self.session, self.session_id = self.session, None, None
        if session is not None:
            session.close(reason)

    def _watchdog(self):
        while True:
            time.sleep(1.0)
            if self.session is not None and time.monotonic() - self.last_call > self.idle_timeout:
                with self.lock:
                    if self.session is not None and time.monotonic() - self.last_call > self.idle_timeout:
                        print(f'session idle for {self.idle_timeout:.0f} s: closed', flush=True)
                        self._close(f'no call for {self.idle_timeout:.0f} s (the client went away)')


def serve(service, port):
    """Start the gRPC server on ``port`` (all interfaces); returns it (``.stop(grace)`` to end)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[('grpc.max_receive_message_length', MAX_MESSAGE),
                                  ('grpc.max_send_message_length', MAX_MESSAGE)])
    rpc.add_RolloutServicer_to_server(service, server)
    if not server.add_insecure_port(f'[::]:{int(port)}'):
        raise RuntimeError(f'cannot listen on port {port}')
    server.start()
    return server


@contextmanager
def _status(context):
    """Session errors -> gRPC status codes (context.abort raises)."""
    try:
        yield
    except UnknownSession as error:
        context.abort(grpc.StatusCode.NOT_FOUND, str(error.args[0]))
    except (ValueError, KeyError) as error:                       # a malformed request or action
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
    except Busy as error:
        context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
    except Stopped as error:
        context.abort(grpc.StatusCode.ABORTED, str(error))
    except (NotReady, RobotError) as error:
        context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
    except Exception as error:
        traceback.print_exc()
        context.abort(grpc.StatusCode.INTERNAL, f'{type(error).__name__}: {error}')
