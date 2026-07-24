#!/bin/bash
# Combined entrypoint: starts AWSIM (domain 0, background) + Autoware (domain 1, foreground)
# in one container. AWSIM's built-in multi-domain bridge handles cross-domain communication.
# Usage: HEADLESS=1 make combined

set -e

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
vehicles="${1:-1}"

# Auto-detect headless mode
if [ -z "${HEADLESS}" ]; then
    if [ -n "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
        HEADLESS=0
    else
        HEADLESS=1
    fi
fi

echo "[INFO] Starting AWSIM on domain 0 (headless=${HEADLESS}, vehicles=${vehicles})..."

log_dir="${LOG_DIR:-/output}"
mkdir -p "${log_dir}"
AWSIM_LOG="${log_dir}/awsim.log"

# ── 1. Start AWSIM on domain 0 (background) ──────────────────────
export ROS_DOMAIN_ID=0

$AWSIM_DIRECTORY/AWSIM.x86_64 \
    --start-mode count \
    --start-count-seconds 0 \
    --vehicles "${vehicles}" \
    --npcs 0 \
    --boosts 2 \
    --laps unlimited \
    --timeout unlimited \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery on \
    --ranking off \
    --camera off \
    --lidar off \
    ${HEADLESS:+--headless} \
    > "$AWSIM_LOG" 2>&1 &

AWSIM_PID=$!
echo "[INFO] AWSIM started (PID=$AWSIM_PID). Log: $AWSIM_LOG"

# Give AWSIM time to initialize its multi-domain bridge
sleep 5

if ! kill -0 $AWSIM_PID 2>/dev/null; then
    echo "[ERROR] AWSIM exited prematurely! Check log: $AWSIM_LOG"
    tail -20 "$AWSIM_LOG"
    exit 1
fi

# ── 2. Start Autoware on domain 1 (foreground) ────────────────────
echo "[INFO] Starting Autoware on domain 1..."
export ROS_DOMAIN_ID=1

set -m
/aichallenge/run_autoware.bash awsim-no-viz 1 /output &
AUTOWARE_PID=$!

trap 'kill -INT $AWSIM_PID $AUTOWARE_PID 2>/dev/null' TERM INT

# Wait for Autoware to finish
while kill -0 $AUTOWARE_PID 2>/dev/null; do wait; done

# Cleanup AWSIM on exit
kill -INT $AWSIM_PID 2>/dev/null
wait $AWSIM_PID 2>/dev/null
echo "[INFO] Combined container shutting down."

