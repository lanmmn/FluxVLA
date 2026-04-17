# FluxVLA Serving

ZMQ-based VLA model serving for remote robot inference.

## Architecture

```
serving/
├── serve.py          # CLI entry point: load model + start server
├── zmq_server.py     # PolicyServer (ZMQ event loop) + create_server factory
├── serializers.py    # Wire-format encode/decode (shared by client & server)
└── proto/
    ├── vla_service.proto      # Protobuf schema
    └── vla_service_pb2.py     # Generated Python code
```

Two-layer design:

```
PolicyServer        Layer 1: generic ZMQ REP loop + endpoint routing
    |
create_server()     Layer 2: wires VLA model into predict_action handler
                    (deserialize obs -> preprocess -> inference ->
                     denormalize -> serialize action)
```

## Quick Start

### Start the server (GPU machine)

```bash
python -m fluxvla.engines.runners.serving.serve \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 5555 \
    --device cuda:0 --dtype bf16
```

### Connect from robot (client)

The client side is `RemoteInferenceRunner` (in `runners/remote_inference_runner.py`),
used via config:

```python
inference = dict(
    type='RemoteURInferenceRunner',
    server_host='192.168.50.125',
    server_port=5555,
    serializer='protobuf',
    compress=True,
    # ...
)
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | Path to mmengine config file |
| `--ckpt-path` | (required) | Path to model checkpoint (.pt / .safetensors) |
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` = all interfaces) |
| `--port` | `5555` | ZMQ TCP port |
| `--device` | `cuda:0` | CUDA device |
| `--dtype` | `bf16` | Mixed precision: `bf16` / `fp16` / `fp32` |
| `--dataset-key` | auto-detect | Config key for dataset pipeline: `inference` or `eval` |

## Wire Formats

Two serialization formats are supported. The first byte of every ZMQ message
determines the format:

| Format | First byte | Pros | Cons |
|--------|-----------|------|------|
| msgpack | != `0x01` | Flexible, zero schema, Python-native | Larger payload |
| protobuf | `0x01` | Compact, cross-language (C++/Rust/Go) | Requires `.proto` schema |

Both formats use the same payload encoding for observation data:
- RGB images: JPEG-compressed bytes (configurable via `compress` flag)
- Numeric arrays: numpy `.npy` format bytes
- Strings: passed directly

## Endpoints

| Endpoint | Input | Output | Description |
|----------|-------|--------|-------------|
| `predict_action` | obs + unnorm_key | action bytes + infer_time | Model inference |
| `ping` | (none) | `{status: ok}` | Health check |
| `reset` | (none) | `{status: ok}` | Reset server state |
| `get_status` | (none) | uptime, request count, avg latency | Server stats |
| `kill` | (none) | `{status: ok}` | Graceful shutdown |

## Data Flow

```
Client (robot)                    Server (GPU)
     |                                 |
     |  encode obs (JPEG + npy)        |
     |  ────── ZMQ REQ ──────────────> |
     |                                 |  deserialize obs
     |                                 |  dataset transform (preprocess)
     |                                 |  vla.predict_action (GPU)
     |                                 |  denormalize action
     |                                 |  serialize action (npy)
     |  <────── ZMQ REP ────────────── |
     |  decode action                  |
     |  execute on robot               |
```
