#!/usr/bin/env bash
# Lock Orange Pi 5 (RK3588S) clocks and governors for deterministic low-latency / max-throughput operation.
# Usage: sudo bash scripts/tune_orangepi5.sh [performance|ondemand]
set -euo pipefail

MODE="${1:-performance}"
if [[ "$MODE" != "performance" && "$MODE" != "ondemand" ]]; then
  echo "Usage: $0 [performance|ondemand]" >&2
  exit 1
fi

echo "[TUNE] Setting CPU governors to: $MODE"
for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
  if [[ -w "$g" ]]; then
    echo "$MODE" > "$g"
  fi
done

NPU_GOV="/sys/class/devfreq/fdab0000.npu/governor"
if [[ -w "$NPU_GOV" ]]; then
  if [[ "$MODE" == "performance" ]]; then
    echo "performance" > "$NPU_GOV"
  else
    echo "rknpu_ondemand" > "$NPU_GOV"
  fi
fi

echo "[TUNE] Current CPU policy frequencies:"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "  $(basename "$p"): gov=$(cat "$p/scaling_governor") cur=$(cat "$p/scaling_cur_freq") kHz"
done
echo "[TUNE] NPU devfreq: gov=$(cat "$NPU_GOV") cur=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq) Hz"
