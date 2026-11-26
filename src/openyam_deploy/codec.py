"""numpy <-> the Rollout wire (proto/rollout.proto), by omniact-molmospaces' rules (molmo_spaces/rollout/codec.py): a
Tensor is raw little-endian bytes + dtype name + shape; an observation's ``node_names`` / ``camera_names`` / ``prompt``
go to their own fields and every other entry to ``tensors``; ``None`` values are dropped; an info scalar maps bool /
int / float to b / i / f and anything else to its string."""
import numpy as np

from .proto import rollout_pb2 as pb

DTYPES = {'float32', 'float64', 'int32', 'int64', 'uint8', 'bool'}
OWN_FIELDS = ('node_names', 'camera_names', 'prompt')


def encode(value):
    array = np.ascontiguousarray(value)
    if array.dtype.name not in DTYPES:
        raise ValueError(f'unsupported tensor dtype {array.dtype}')
    return pb.Tensor(shape=array.shape, dtype=array.dtype.name,
                     data=array.astype(array.dtype.newbyteorder('<')).tobytes())


def decode(tensor):
    if tensor.dtype not in DTYPES or len(tensor.shape) > 8 or min(tensor.shape, default=0) < 0:
        raise ValueError(f'unsupported tensor {tensor.dtype} {list(tensor.shape)}')
    dtype = np.dtype(tensor.dtype).newbyteorder('<')
    if int(np.prod(tensor.shape)) * dtype.itemsize != len(tensor.data):
        raise ValueError('tensor byte count does not match its shape')
    return np.frombuffer(tensor.data, dtype).reshape(tuple(tensor.shape)).astype(dtype.newbyteorder('='))


def observation(obs):
    return pb.Observation(tensors={k: encode(v) for k, v in obs.items() if k not in OWN_FIELDS and v is not None},
                          node_names=obs.get('node_names') or [], camera_names=obs.get('camera_names') or [],
                          prompt=obs.get('prompt') or '')


def info(values):
    out = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            out[key] = pb.Value(b=bool(value))
        elif isinstance(value, (int, np.integer)):
            out[key] = pb.Value(i=int(value))
        elif isinstance(value, (float, np.floating)):
            out[key] = pb.Value(f=float(value))
        else:
            out[key] = pb.Value(s=str(value))
    return out
