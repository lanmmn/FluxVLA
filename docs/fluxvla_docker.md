
## Step 0: 准备工作

### 0.1 挂载 NVMe SSD（可选但强烈建议）

eMMC 仅剩 35GB，模型和数据集会很快占满。

```bash
# 分区
sudo fdisk /dev/nvme0n1
# 交互输入: n → 回车 → 回车 → 回车 → 回车 → w

# 格式化
sudo mkfs.ext4 /dev/nvme0n1p1

# 挂载
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
sudo chown limx:limx /mnt/nvme

# 开机自动挂载
echo '/dev/nvme0n1p1 /mnt/nvme ext4 defaults 0 2' | sudo tee -a /etc/fstab

# 创建工作目录
mkdir -p /mnt/nvme/{checkpoints,datasets,work_dirs}
```



### 0.2 配置 pip 国内源

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### 0.3 安装系统依赖

```bash
sudo apt update
sudo apt install -y cmake libglew-dev libosmesa6-dev ninja-build
```
