"""The Rollout wire contract, vendored from omniact-molmospaces (molmo_spaces/rollout/proto/: rollout.proto and its
generated rollout_pb2.py / rollout_pb2_grpc.py, protobuf 6.31.1 / grpcio 1.76.0, the versions pyproject.toml pins).
The only changes: rollout_pb2_grpc.py imports rollout_pb2 relatively, and rollout.proto's header comment names the
omniact client. Being the same messages and service (rollout.Rollout), this server and omniact-molmospaces'
eval_server are interchangeable to OmniAct's client."""
