# FluxVLA Remote 模块架构

## 文件概览

| 文件               | 职责                                                         | 角色       |
| ------------------ | ------------------------------------------------------------ | ---------- |
| `serve.py`         | CLI 入口，加载模型/配置，启动服务器                          | 启动脚本   |
| `vla_server.py`    | 模型推理流水线 + 策略包装 + 服务器工厂                       | 服务端核心 |
| `server_client.py` | ZMQ 通信框架（PolicyServer / ObsSerializer / MsgSerializer） | 通信层     |
| `serializers.py`   | protobuf / msgpack 双格式编解码                              | 序列化层   |
| `remote_vla.py`    | 客户端代理，伪装成本地 VLA 模型                              | 客户端核心 |

## 调用关系

```mermaid
graph TB
    subgraph "启动脚本"
        serve["serve.py<br/><i>CLI 入口</i><br/>加载 config / checkpoint<br/>构建 dataset pipeline"]
    end

    subgraph "服务端"
        vla_server["vla_server.py"]
        VLAInferPipeline["VLAInferPipeline<br/><i>模型推理</i><br/>autocast + predict_action"]
        VLAPolicy["VLAPolicy(BasePolicy)<br/><i>请求处理</i><br/>反序列化 obs → 预处理 →<br/>推理 → 反归一化 → 序列化 action"]
        create_vla_server["create_vla_server()<br/><i>工厂函数</i>"]
    end

    subgraph "通信层"
        server_client["server_client.py"]
        PolicyServer["PolicyServer<br/><i>ZMQ REP 服务器</i><br/>接收请求 → 路由 endpoint →<br/>返回结果"]
        ObsSerializer["ObsSerializer<br/><i>msgpack 观测序列化</i><br/>JPEG 压缩图像 + npy 数组"]
        MsgSerializer["MsgSerializer<br/><i>msgpack 通用序列化</i><br/>ping / kill / get_status"]
    end

    subgraph "序列化层"
        serializers["serializers.py"]
        ObsSerializerProto["ObsSerializerProto<br/><i>protobuf 观测序列化</i>"]
        encode_req["encode_predict_request()<br/>obs dict → bytes"]
        decode_req["decode_predict_request()<br/>bytes → obs dict"]
        encode_resp["encode_predict_response()<br/>action bytes → bytes"]
        decode_resp["decode_predict_response()<br/>bytes → action dict"]
    end

    subgraph "客户端"
        remote_vla["remote_vla.py"]
        RemoteVLAZmq["RemoteVLAZmq<br/><i>客户端代理</i><br/>duck-typing 兼容本地 VLA<br/>predict_action / ping / close"]
    end

    serve -->|"构建"| VLAInferPipeline
    serve -->|"调用"| create_vla_server
    create_vla_server -->|"创建"| VLAPolicy
    create_vla_server -->|"创建"| PolicyServer
    VLAPolicy -->|"调用推理"| VLAInferPipeline
    VLAPolicy -->|"解码 obs"| ObsSerializer
    VLAPolicy -->|"解码 obs"| ObsSerializerProto
    PolicyServer -->|"路由请求到"| VLAPolicy
    PolicyServer -->|"通用序列化"| MsgSerializer
    PolicyServer -->|"protobuf 解码"| decode_req

    RemoteVLAZmq -->|"编码请求"| encode_req
    RemoteVLAZmq -->|"解码响应"| decode_resp
    encode_req -->|"msgpack 路径"| ObsSerializer
    encode_req -->|"protobuf 路径"| ObsSerializerProto

    RemoteVLAZmq -.->|"ZMQ TCP"| PolicyServer
```

## 数据流：一次 predict_action 请求

```mermaid
sequenceDiagram
    participant Runner as InferenceRunner
    participant Client as RemoteVLAZmq<br/>(remote_vla.py)
    participant Ser as serializers.py
    participant Net as ZMQ TCP
    participant Server as PolicyServer<br/>(server_client.py)
    participant Policy as VLAPolicy<br/>(vla_server.py)
    participant Model as VLAInferPipeline

    Runner->>Client: predict_action(images, proprio, task_desc)

    Note over Client: torch.Tensor → numpy
    Client->>Ser: encode_predict_request(obs, unnorm_key, fmt, compress)

    alt protobuf
        Ser->>Ser: ObsSerializerProto.obs_to_proto()
        Note over Ser: 图像 → JPEG bytes (或 npy)<br/>数组 → npy bytes
    else msgpack
        Ser->>Ser: ObsSerializer.to_bytes()
        Note over Ser: 图像 → JPEG bytes (或 npy)<br/>数组 → npy bytes
    end

    Ser-->>Client: request bytes (~30KB)
    Client->>Net: socket.send(request)
    Net->>Server: socket.recv()

    alt protobuf (第一字节 == 0x01)
        Server->>Ser: decode_predict_request()
        Ser-->>Server: obs dict + unnorm_key
        Server->>Policy: predict_action(_obs_dict=obs)
    else msgpack
        Server->>Server: MsgSerializer.from_bytes()
        Server->>Policy: predict_action(obs_data=bytes)
        Policy->>Policy: ObsSerializer.from_bytes()
    end

    Note over Policy: dataset transforms:<br/>resize, normalize, tokenize
    Policy->>Model: predict_action(**batch)
    Note over Model: GPU forward pass<br/>(autocast bf16)
    Model-->>Policy: action tensor
    Note over Policy: denormalize action
    Policy->>Policy: TensorSerializer.serialize_actions()

    Policy-->>Server: {action_data, infer_time}
    Server->>Ser: encode_predict_response()
    Ser-->>Server: response bytes
    Server->>Net: socket.send(response)
    Net->>Client: socket.recv()

    Client->>Ser: decode_predict_response()
    Ser-->>Client: {action_data, infer_time}
    Note over Client: np.load → torch.Tensor<br/>→ self._device
    Client-->>Runner: action tensor (1, T, D)
```

## 文件命名与职责对照

### 当前命名的困惑点

| 困惑                                                        | 原因                                                        |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| `serve.py` vs `vla_server.py`                               | 都像"服务端"，但一个是 CLI 入口，一个是业务逻辑             |
| `server_client.py`                                          | 名字暗示同时包含 server 和 client，实际主要是 server 通信层 |
| `serializers.py` vs `server_client.py` 中的 `ObsSerializer` | 两个文件都有序列化逻辑，职责重叠                            |
| `remote_vla.py`                                             | 名字不够直观，实际是"客户端代理"                            |

### 各文件的实际定位

```
fluxvla/remote/
├── serve.py            # 入口脚本（等价于 main.py）
│                       # 解析 CLI 参数 → 加载模型 → 构建 pipeline → 启动 server
│
├── vla_server.py       # 服务端业务逻辑
│                       # VLAInferPipeline: 模型推理封装
│                       # VLAPolicy: obs 解码 → 预处理 → 推理 → 反归一化
│                       # create_vla_server(): 工厂函数
│
├── server_client.py    # ZMQ 通信框架（与业务无关的通用层）
│                       # PolicyServer: ZMQ REP 事件循环 + endpoint 路由
│                       # MsgSerializer: msgpack 通用序列化
│                       # ObsSerializer: msgpack 观测序列化（JPEG + npy）
│
├── serializers.py      # 双格式序列化（protobuf + msgpack 的统一入口）
│                       # ObsSerializerProto: protobuf 观测序列化
│                       # encode/decode_predict_request/response
│                       # 格式自动检测（第一字节 0x01 = protobuf）
│
├── remote_vla.py       # 客户端代理（伪装成本地 VLA 模型）
│                       # RemoteVLAZmq: predict_action → ZMQ → server
│                       # duck-typing: eval/cuda/freeze_* 兼容 runner
│
└── proto/
    ├── vla_service.proto    # protobuf schema 定义
    └── vla_service_pb2.py   # protoc 生成的 Python 代码
```
