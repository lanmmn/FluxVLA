# Orin FluxVLA：装机与配置

更新时间：2026-07-27<br>
设备：Jetson AGX Orin（主机名 `orin`）

本文只记录基础环境如何搭建和配置。项目改动与验证记录见
`orin_02_work_record.md`，现场启动命令见 `orin_03_launch_guide.md`。

## 1. 最终环境

```text
USB 管理地址        192.168.55.1
USB 对端地址        192.168.55.100
SSH 用户            limx
数据盘              /data == /mnt/nvme
FluxVLA 仓库        /data/FluxVLA
兼容软链接          /home/limx/sober/FluxVLA -> /data/FluxVLA
checkpoint 根目录   /data/ckpts
Docker 镜像         fluxvla:orin-ros-fa
常驻容器            fluxvla_eval
Docker 数据目录     /data/docker-data
```

现场网络拓扑：

```text
操作电脑 --USB--> Orin 192.168.55.1       管理与 SSH
Orin     --网线--> 机器人网络 10.192.1.x   MROS、相机和机器人状态
```

## 2. 系统与 NVMe

系统底座：

```text
L4T      R36.4.4
Ubuntu   22.04.5
内核     5.15.148-tegra
```

1 TB Samsung NVMe 使用 GPT 单分区和 ext4，挂载到 `/data`。`/etc/fstab`
中的持久化配置为：

```fstab
UUID=5b605aa8-dc25-43c7-8d72-bd0bd3120c61 /data ext4 defaults,nofail,x-systemd.device-timeout=10 0 2
/data /mnt/nvme none bind,nofail 0 0
```

验证：

```bash
df -h /data /mnt/nvme
findmnt /data
findmnt /mnt/nvme
```

若重启后未挂载：

```bash
sudo mount -a
```

## 3. USB SSH 与临时联网

连接：

```bash
ssh limx@192.168.55.1
```

换操作电脑时，先确认 USB 网卡：

```bash
ip -brief addr | grep 192.168.55
ssh-copy-id limx@192.168.55.1
```

若 Orin 临时需要访问外网，可在操作电脑上启用 NAT。以下接口名必须按操作电脑
实际情况修改：

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o enp3s0 -j MASQUERADE
sudo iptables -A FORWARD -i enxb212353ba5ba -o enp3s0 -j ACCEPT
sudo iptables -A FORWARD -i enp3s0 -o enxb212353ba5ba \
  -m state --state RELATED,ESTABLISHED -j ACCEPT
```

Orin 使用 `192.168.55.100` 作为 USB 侧网关。DNS 配置位于：

```text
/etc/systemd/resolved.conf.d/upstream.conf
```

NAT 规则在操作电脑重启后需要重新设置。现场真机推理不依赖外网。

## 4. FluxVLA 代码

初始仓库：

```text
https://github.com/lanmmn/FluxVLA.git
分支：feat/lzh/Oli-in-place-develop
初始 commit：4501110
```

安装位置：

```bash
cd /data
git clone --depth 1 --branch feat/lzh/Oli-in-place-develop \
  https://github.com/lanmmn/FluxVLA.git
ln -s /data/FluxVLA /home/limx/sober/FluxVLA
```

当前仓库包含 Orin 适配改动，不应直接用全新 checkout 覆盖。具体改动见工作记录。

## 5. Docker 与 NVIDIA runtime

安装：

```bash
sudo apt-get install -y docker.io nvidia-container-toolkit docker-buildx
sudo usermod -aG docker limx
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
```

Docker 和 containerd 数据已迁到 NVMe：

```text
Docker data-root    /data/docker-data/docker
containerd root     /data/docker-data/containerd
```

验证：

```bash
docker info | grep -i 'Docker Root Dir'
docker images | grep fluxvla
```

## 6. 构建镜像

构建链：

```text
base -> wheel -> fa -> ros -> ros-fa
```

目标镜像：

```text
fluxvla:orin-ros-fa
```

构建命令：

```bash
cd /data/FluxVLA
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh all
```

当前分支构建时已处理过以下问题：

- 补齐 `vendor/dlimp`；
- `pip wheel` 使用 `--no-deps`，避免拉入不兼容的通用 CUDA torch；
- 为 ros-fa 显式传递已编好的 flash-attn wheel；
- Docker 与 containerd 数据迁到 NVMe，避免 eMMC 被镜像写满；
- CUDA 自定义算子不在基础镜像中，需在源码挂载目录单独编译。

构建日志：

```text
/data/build-logs/
```

## 7. 外部依赖与 CUDA 扩展

机器人运行环境：

```text
/opt/limx/robot-tron2-r/install/
```

MROS wheel：

```text
/home/limx/mros-2.3.1-py3-none-any.whl
```

必须使用 aarch64 版本。x86-64 的 `mrostopic` 等工具不能直接在 Orin 上运行。

CUDA 扩展在容器内编译，产物写入持久化源码目录：

```bash
cd /workspace/FluxVLA
FORCE_CUDA=1 MAX_JOBS=4 TMPDIR=/data/tmp \
  python3 setup.py build_ext \
  --build-temp /data/tmp/build_ext --inplace
```

应生成：

```text
gemma_rotary_embedding_ext
rotary_pos_embedding_ext
matmul_bias_ext
```

## 8. 创建常驻容器

只有容器丢失时才需要重新创建：

```bash
docker run -d --name fluxvla_eval \
  --runtime=nvidia --ipc=host --network=host --shm-size=16g \
  -e PYTHONPATH=/workspace/FluxVLA:/opt/limx/robot-tron2-r/install/bin/mrosrs/src \
  -e MROS_IP_LIST=127.0.0.1 \
  -e WANDB_MODE=disabled \
  -e ATTN_IMPLEMENTATION=flash_attention_2 \
  -e TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2 \
  -v /data/FluxVLA:/workspace/FluxVLA \
  -v /data:/data \
  -v /mnt/nvme:/mnt/nvme \
  -v /opt/limx:/opt/limx:ro \
  -v /home/limx/mros-2.3.1-py3-none-any.whl:/tmp/mros-2.3.1-py3-none-any.whl:ro \
  -w /workspace/FluxVLA \
  fluxvla:orin-ros-fa sleep infinity
```

容器重建后安装 MROS：

```bash
docker exec fluxvla_eval bash -lc '
source /opt/limx/robot-tron2-r/install/setup.bash
python3 -m pip install /tmp/mros-2.3.1-py3-none-any.whl
python3 -c "import mros; print(\"mros ok\")"
'
```

源码、checkpoint 和 CUDA `.so` 在 `/data`，删除容器不会删除这些文件；容器内
单独安装的 Python 包会随容器删除。

## 9. 性能模式

推理使用 MAXN 和锁频：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

当前已配置：

```text
nvpmodel          MAXN(0)
GPU               1300.5 MHz
jetson_clocks     systemd 开机自启
service           /etc/systemd/system/jetson-clocks.service
```

重新配置 MAXN 会立即重启设备：

```bash
yes | sudo nvpmodel -m 0
sudo jetson_clocks
sudo systemctl enable jetson-clocks.service
```

满频运行时应使用 `tegrastats` 监控温度，避免 thermal throttle。

## 10. 基础健康检查

```bash
df -h /data
docker images | grep fluxvla
docker start fluxvla_eval
docker exec fluxvla_eval bash -lc '
python3 -c "import torch, flash_attn, mros; print(torch.cuda.is_available())"
'
```

这里仅检查环境，不启动模型或机器人动作。正式运行请使用启动手册。
