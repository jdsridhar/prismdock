#!/usr/bin/env bash
set -e
export PATH="/home/sridhar/miniconda3/envs/cadd/bin:/home/sridhar/jre17/bin:$PATH"
export JAVA_HOME="/home/sridhar/jre17"
export P2RANK_HOME="/mnt/c/Users/sridhar/OneDrive/Documents/tagore_dock/vendor/p2rank"

cd /mnt/c/Users/sridhar/OneDrive/Documents/tagore_dock

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: ./run_wsl_pipeline.sh <config.yaml>"
    exit 1
fi

echo "Running PrismDock Pipeline with config: $CONFIG"
/home/sridhar/miniconda3/envs/cadd/bin/python run_dockathon.py --config "$CONFIG"
