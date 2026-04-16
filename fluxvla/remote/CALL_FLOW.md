# FluxVLA Remote Inference - Call Flow

## 文件依赖关系

```
policy.py           <- 抽象基类,定义 BasePolicy 接口
    ^
server_client.py    <- 通信框架层 (MsgSerializer, ObsSerializer, PolicyServer, PolicyClient)
    ^
vla_server.py       <- Server 端推理封装 (VLAInferPipeline, VLAPolicy, TensorSerializer)
remote_vla.py       <- Client 端代理 (RemoteVLAZmq)
```

## 1. Server 启动流程

入口: `fluxvla/remote/serve.py -> main()` (或 `python -m fluxvla.remote.serve`)

```
serve.py main()
|
|- 1. Config.fromfile(args.config)                    # 加载 mmengine config
|- 2. build_vla_from_cfg(cfg.model)                   # 构建 VLA 模型 (PI05FlowMatching)
|- 3. torch.load(ckpt_path) -> vla.load_state_dict()  # 加载模型权重
|- 4. json.load(dataset_statistics.json)              # 加载归一化统计
|     -> vla.norm_stats = ...
|- 5. build_dataset_from_cfg(cfg.eval.dataset)        # 构建预处理管道
|- 6. build_transform_from_cfg(cfg.eval.denormalize_action)  # 构建反归一化 transform
|
|- 7. VLAInferPipeline(vla, device, dtype)       <- vla_server.py
|     '- vla.eval().to(device)                        # 模型移到 GPU, 设为推理模式
|
|- 8. create_vla_server(pipeline, dataset, ...)  <- vla_server.py
|     |- VLAPolicy(pipeline, dataset, denormalize_action, task_suite_name)
|     |   '- 继承 BasePolicy                    <- policy.py
|     |- PolicyServer(policy, host, port)        <- server_client.py
|     |   |- zmq.Context()                            # 创建 ZMQ 上下文
|     |   |- socket(zmq.REP)                          # 创建 REP socket
|     |   |- socket.bind(tcp://host:port)             # 绑定端口
|     |   '- register_endpoint("predict_action", ...)  # 注册端点
|     '- return server
|
'- 9. server.run()                               <- server_client.py PolicyServer.run()
      '- while self.running:                          # 主事件循环
           |- poller.poll(timeout=500)                # 非阻塞等待请求
           |- socket.recv()                           # 接收原始字节
           |- MsgSerializer.from_bytes(message)       # msgpack 反序列化
           |- endpoint = request["endpoint"]          # 路由到对应 handler
           |- handler(**request["data"])               # 调用 handler (如 predict_action)
           '- socket.send(MsgSerializer.to_bytes(result))  # 返回结果
```

## 2. 一次推理请求 (Client -> Server -> Client)

### Client 端

入口: `LiberoEvalRunner.run()` -> `self.vla.predict_action(**obs)`

```
RemoteVLAZmq.predict_action(**kwargs)         <- remote_vla.py
|
|- 1. kwargs.pop("unnorm_key")                      # 分离非观测字段
|- 2. tensor -> numpy                                # 所有 torch.Tensor 转为 numpy
|- 3. ObsSerializer.to_bytes(obs)              <- server_client.py
|     |- 图像 (HWC uint8) -> cv2.imencode JPEG       # JPEG 压缩 (~10x)
|     |- 其余 ndarray -> np.save -> bytes              # npy 格式
|     '- msgpack.packb(encoded)                      # 合并打包
|
|- 4. msgpack.packb({endpoint, data})                # 构造请求
|- 5. socket.send(request)                           # ZMQ 发送
|- 6. socket.recv()                                  # 阻塞等待响应
|- 7. msgpack.unpackb(raw_response)                  # 解包响应
|
|- 8. torch.load(response["action_data"],            # 反序列化 action
|                map_location=device,
|                weights_only=True)
|
'- 9. return actions                                 # 返回已反归一化的 action tensor
```

### Server 端

Client 的 step 5-6 期间, Server 内部执行:

```
PolicyServer.run() 的事件循环收到请求
|
'- VLAPolicy.predict_action(obs_data, unnorm_key)  <- vla_server.py
   |
   |- 1. ObsSerializer.from_bytes(obs_data)      <- server_client.py
   |     |- msgpack.unpackb(data)                      # 解包
   |     |- JPEG bytes -> cv2.imdecode -> numpy          # JPEG 解码还原图像
   |     '- npy bytes -> np.load -> numpy                # 还原数组
   |
   |- 2. self._dataset(obs)                            # 预处理管道
   |     |- ProcessLiberoEvalInputs                    # 图像加载/裁切
   |     |- TransformImage                             # resize + normalize
   |     |- LiberoPromptFromInputs                     # tokenize task description
   |     '- LiberoProprioFromInputs                    # 状态归一化
   |
   |- 3. VLAInferPipeline.predict_action(**batch)  <- vla_server.py
   |     |- batch tensors -> .to(device)                # 移到 GPU
   |     '- torch.autocast -> vla.predict_action()      # 模型前向推理
   |         '- PI05FlowMatching.predict_action()      # flow matching 采样
   |
   |- 4. self._denormalize_action(...)                 # 反归一化
   |     '- DenormalizeLiberoAction.__call__()     <- transforms/normalize.py
   |         |- action * std + mean                    # 还原到真实物理量
   |         |- normalize_gripper_action()             # gripper 二值化
   |         |- invert_gripper_action()                # gripper 翻转
   |         '- action[:action_dim]                    # 截断到 7 维
   |
   |- 5. TensorSerializer.serialize_actions(actions)   # torch.save -> bytes
   |
   '- 6. return {"action_data": bytes, "infer_time": float}
         -> PolicyServer 将其 MsgSerializer.to_bytes() 后发回 Client
```

## 总结: 一图看全

```
Client (机器人/仿真)                      Server (GPU)
===================                      ===========
env.step() -> obs                         PolicyServer.run() 事件循环等待
     |                                         ^
RemoteVLAZmq                              VLAPolicy
     |                                         |
 ObsSerializer.to_bytes()                  ObsSerializer.from_bytes()
     |           -- ZMQ REQ -->                |
     |                                    dataset(obs)        预处理
     |                                         |
     |                                    VLAInferPipeline    GPU推理
     |                                         |
     |                                    denormalize_action  反归一化
     |                                         |
     |           <-- ZMQ REP --           TensorSerializer
     |
 torch.load(action)
     |
 action.tolist() -> env.step()
```

## 各文件职责速查

| 文件               | 角色        | 核心类/函数                                                                |
| ------------------ | ----------- | -------------------------------------------------------------------------- |
| `policy.py`        | 抽象接口    | `BasePolicy` (ABC)                                                         |
| `server_client.py` | 通信框架    | `MsgSerializer`, `ObsSerializer`, `PolicyServer`, `PolicyClient`           |
| `vla_server.py`    | Server 推理 | `VLAInferPipeline`, `VLAPolicy`, `TensorSerializer`, `create_vla_server()` |
| `remote_vla.py`    | Client 代理 | `RemoteVLAZmq`                                                             |
| `__init__.py`      | 包导出      | 汇总导出所有公开类                                                         |

## 序列化协议栈

```
                       Client 发送                Server 接收
                       ==========                ==========

应用层     raw obs dict (numpy + str)        raw obs dict (numpy + str)
               |                                    ^
序列化层   ObsSerializer.to_bytes()          ObsSerializer.from_bytes()
           |- 图像: JPEG 压缩                |- JPEG: cv2.imdecode
           |- 数组: np.save (npy)            |- npy: np.load
           '- 合并: msgpack.packb            '- 拆分: msgpack.unpackb
               |                                    ^
消息层     msgpack.packb({                   msgpack.unpackb(message)
             endpoint, data                      -> 路由到 handler
           })                                       |
               |                                    ^
传输层     zmq.REQ.send()  ---- TCP ----  zmq.REP.recv()


                       Server 返回                Client 接收
                       ==========                ==========

应用层     action tensor (denormed)          action tensor (on device)
               |                                    ^
序列化层   TensorSerializer                  torch.load(buf,
             .serialize_actions()              map_location=device,
           = torch.save(cpu tensor)            weights_only=True)
               |                                    ^
消息层     MsgSerializer.to_bytes({          msgpack.unpackb(raw)
             action_data, infer_time           -> response dict
           })                                       |
               |                                    ^
传输层     zmq.REP.send()  ---- TCP ----  zmq.REQ.recv()
```
