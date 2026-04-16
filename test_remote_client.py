"""本地测试远程推理 client，通过 SSH 隧道连接 GPU server。

使用前先建立 SSH 隧道:
    ssh -p 57705 -L 5555:127.0.0.1:3333 user@14.103.233.39

然后运行:
    python test_remote_client.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, '.')

from fluxvla.remote import RemoteVLAZmq

# SSH 隧道映射: 本地 5555 → 远程 3333
HOST = '127.0.0.1'
PORT = 5555
DEVICE = 'cpu'  # Mac 无 GPU，action 反序列化放 CPU


def main():
    # 1. 连接 server
    print(f'Connecting to {HOST}:{PORT} ...')
    vla = RemoteVLAZmq(host=HOST, port=PORT, timeout_s=30.0, device=DEVICE)

    # 2. 健康检查
    print('Pinging server ...')
    if not vla.ping():
        print('FAILED: server unreachable, check SSH tunnel and server status')
        vla.close()
        sys.exit(1)
    print('Server is alive!')

    # 3. 构造 UR3 风格 raw observation
    obs = {
        'cam_high': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'cam_left_wrist':
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'qpos': np.random.randn(7).astype(np.float32),
        'task_description': 'grasp the bottle',
        'unnorm_key': 'private',
    }

    # 4. 推理
    print('Sending inference request ...')
    t0 = time.perf_counter()
    actions = vla.predict_action(**obs)
    elapsed = (time.perf_counter() - t0) * 1000

    # 5. 结果
    print(f'Action shape: {actions.shape}')
    print(f'Action dtype: {actions.dtype}')
    print(f'Action sample: {actions[0, 0, :7].cpu().numpy()}')
    print(f'Client total: {elapsed:.1f} ms')

    # 详细 profiling
    p = vla._last_profile
    print(f'  serialize:    {p["serialize_ms"]:.1f} ms')
    print(f'  zmq roundtrip:{p["zmq_roundtrip_ms"]:.1f} ms')
    print(f'  server infer: {p["server_infer_ms"]:.1f} ms')
    print(f'  network:      {p["network_ms"]:.1f} ms')
    print(f'  deserialize:  {p["deserialize_ms"]:.1f} ms')
    print(f'  payload:      {p["payload_kb"]:.0f} KB')

    # 6. 多次推理验证稳定性
    print('\nRunning 5 more requests ...')
    for i in range(5):
        actions = vla.predict_action(**obs)
        p = vla._last_profile
        print(f'  [{i+1}] total={p["total_ms"]:.1f}ms  '
              f'server={p["server_infer_ms"]:.1f}ms  '
              f'shape={actions.shape}')

    vla.close()
    print('\nDone!')


if __name__ == '__main__':
    main()
