# Remote inference config for UR3 + Pi0.5 (ZMQ)
#
# Robot side (no GPU needed):
#   python scripts/inference.py \
#       --config configs/pi05/pi05_paligemma_ur3_remote_inference.py
#
# GPU server side:
#   python -m fluxvla.engines.runners.serving.serve \
#       --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
#       --ckpt-path /path/to/checkpoint.pt \
#       --host 0.0.0.0 --port 5555

inference = dict(
    type='RemoteURInferenceRunner',
    server_host='192.168.50.125',
    server_port=5555,
    timeout_s=30.0,
    serializer='protobuf',
    compress=True,
    seed=7,
    action_chunk=10,
    publish_rate=30,
    max_publish_step=10000,
    task_suite_name='private',
    state_dim=7,
)
