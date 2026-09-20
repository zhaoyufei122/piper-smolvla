#!/usr/bin/env bash
# 配置 3Dconnexion SpaceMouse 免 sudo 权限规则 (udev rule)

set -e

RULE_FILE="/etc/udev/rules.d/99-spacemouse.rules"

echo "=== 正在配置 3Dconnexion SpaceMouse udev 规则 ==="

sudo bash -c "cat << 'EOF' > ${RULE_FILE}
# 3Dconnexion SpaceMouse 权限规则 (VID: 0x256f)
SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"256f\", MODE=\"0666\", GROUP=\"plugdev\"
KERNEL==\"hidraw*\", ATTRS{idVendor}==\"256f\", MODE=\"0666\", GROUP=\"plugdev\"
KERNEL==\"event*\", ATTRS{idVendor}==\"256f\", MODE=\"0666\", GROUP=\"plugdev\"
EOF"

echo "[✓] 已写入规则到 ${RULE_FILE}"

echo "=== 正在重载 udev 规则 ==="
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "[✓] udev 规则生效完成！"
echo "提示：如果当前终端仍无法直接读取，请将 SpaceMouse 的 USB 线重新拔插一次。"
