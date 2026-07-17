#!/bin/bash

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

# 車両数: 第1引数（既定 1）
vehicles="${1:-1}"

# GPUがない場合 -headlessを末尾に追加
exec $AWSIM_DIRECTORY/AWSIM.x86_64 \
    --start-mode count \
    --start-count-seconds 0 \
    --vehicles "${vehicles}" \
    --npcs 0 \
    --boosts 2 \
    --laps unlimited \
    --timeout 10000000.0 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery on \
    --ranking off \
    --camera off \
    --lidar off \
    ${HEADLESS:+--headless}

