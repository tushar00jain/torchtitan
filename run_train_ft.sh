#!/bin/bash

FT_REPLICA_ID="${FT_REPLICA_ID:-0}"
FT_GROUP_SIZE="${FT_GROUP_SIZE:-1}"

TORCH_SHARE_RDZV_TCP_STORE=1 LOGLEVEL=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="${FT_REPLICA_ID}" NGPU=1 ./run_train.sh \
    --fault_tolerance.enable \
    --fault_tolerance.group_size="${FT_GROUP_SIZE}" \
    --fault_tolerance.replica_id="${FT_REPLICA_ID}" \
    --training.local_batch_size=2 \
    --fault_tolerance.sync_steps=10 \
    --fault_tolerance.semi_sync_method=diloco \
    --parallelism.data_parallel_shard_degree=1 \
    --fault_tolerance.num_fragments=2 \
    --experimental.custom_args_module=torchtitan.components.ft.config \
    --profiling.enable_profiling \
    --profiling.profile_freq=5 \
    --profiling.profiler_active=5 \
    --profiling.profiler_warmup=0 \
    --training.steps=1000 \
    --comm.train_timeout_seconds=1 \
    --fault_tolerance.process_group=nccl \
    --checkpoint.no_enable_ft_checkpointing \
    --fault_tolerance.process_group_timeout_ms=1
