#!/bin/bash

set -ex

export CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model_ft.toml"}

export FT_REPLICA_ID="${FT_REPLICA_ID:-0}"
export FT_GROUP_SIZE="${FT_GROUP_SIZE:-1}"

./run_train.sh \
    --fault_tolerance.group_size="${FT_GROUP_SIZE}" \
    --fault_tolerance.replica_id="${FT_REPLICA_ID}"
