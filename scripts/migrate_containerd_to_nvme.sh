#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-/mnt/nvme/containerd}"
CONFIG="/etc/containerd/config.toml"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: run as root, for example: sudo bash $0" >&2
  exit 1
fi

if [[ ! -d /mnt/nvme ]]; then
  echo "Error: /mnt/nvme does not exist or is not mounted" >&2
  exit 1
fi

echo "== Before =="
df -h / /mnt/nvme || true
docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}' 2>/dev/null || true

echo "== Stop Docker/containerd =="
systemctl stop docker.socket 2>/dev/null || true
systemctl stop docker 2>/dev/null || true
systemctl stop containerd

echo "== Prepare target root: ${TARGET_ROOT} =="
mkdir -p "${TARGET_ROOT}"

if [[ -d /var/lib/containerd && ! -L /var/lib/containerd ]]; then
  echo "== Copy existing /var/lib/containerd data if any =="
  if command -v rsync >/dev/null 2>&1; then
    rsync -aHAXS --numeric-ids /var/lib/containerd/ "${TARGET_ROOT}/"
  else
    cp -a /var/lib/containerd/. "${TARGET_ROOT}/"
  fi
fi

echo "== Backup and update containerd config =="
mkdir -p /etc/containerd
if [[ -f "${CONFIG}" ]]; then
  cp -a "${CONFIG}" "${CONFIG}.bak.${STAMP}"
else
  printf 'disabled_plugins = ["cri"]\n' > "${CONFIG}"
  cp -a "${CONFIG}" "${CONFIG}.bak.${STAMP}"
fi

python3 - "${CONFIG}" "${TARGET_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

config = Path(sys.argv[1])
target = sys.argv[2]
s = config.read_text()

root_line = f'root = "{target}"'
state_line = 'state = "/run/containerd"'

if re.search(r'(?m)^\s*#?\s*root\s*=', s):
    s = re.sub(r'(?m)^\s*#?\s*root\s*=.*$', root_line, s, count=1)
else:
    s = root_line + '\n' + s

if re.search(r'(?m)^\s*#?\s*state\s*=', s):
    s = re.sub(r'(?m)^\s*#?\s*state\s*=.*$', state_line, s, count=1)
else:
    s = state_line + '\n' + s

config.write_text(s)
PY

echo "== New containerd root lines =="
grep -nE '^[[:space:]]*(root|state)[[:space:]]*=' "${CONFIG}"

echo "== Start services =="
systemctl daemon-reload
systemctl start containerd
systemctl start docker

echo "== Verify services =="
systemctl --no-pager --full status containerd docker | sed -n '1,80p'

echo "== Verify container filesystem =="
docker run --rm nvcr.io/nvidia/l4t-jetpack:r36.4.0 sh -lc 'df -h / /usr /var; mount | sed -n "1p"'

echo "== After =="
df -h / /mnt/nvme

echo "Done. If the container df output no longer shows the 57G root partition, rerun the FluxVLA build."
echo "Old config backup: ${CONFIG}.bak.${STAMP}"
