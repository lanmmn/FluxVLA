#!/usr/bin/env bash
set -euo pipefail

USER_HOME="${SUDO_USER:+/home/${SUDO_USER}}"
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
  USER_HOME="${HOME}"
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: run as root, for example: sudo bash $0" >&2
  exit 1
fi

echo "== Before =="
df -h / /mnt/nvme || true

# 1) apt caches and partial lists
echo "== Clean apt cache =="
apt-get clean || true
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /var/cache/apt/*.bin || true

# 2) systemd journals (keep only recent logs)
echo "== Vacuum journal logs =="
journalctl --vacuum-time=3d || true
journalctl --vacuum-size=200M || true

# 3) tmp directories
echo "== Clean tmp directories =="
rm -rf /tmp/* /var/tmp/* || true

# 4) old VS Code server versions (keep active one)
if [[ -d "${USER_HOME}/.vscode-server/cli/servers" ]]; then
  echo "== Remove old VS Code server versions =="
  active="$(ps -eo args | grep -F '.vscode-server/cli/servers/Stable-' | grep -v grep | head -1 | sed -n 's#.*cli/servers/\(Stable-[^/]*\)/.*#\1#p')"
  while IFS= read -r d; do
    b="$(basename "$d")"
    if [[ -n "${active}" && "${b}" == "${active}" ]]; then
      continue
    fi
    rm -rf "$d" || true
  done < <(find "${USER_HOME}/.vscode-server/cli/servers" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
fi

# 5) VSIX cache
echo "== Remove VSIX cache =="
rm -rf "${USER_HOME}/.vscode-server/data/CachedExtensionVSIXs" || true

echo "== After =="
df -h / /mnt/nvme || true

echo "Done."
