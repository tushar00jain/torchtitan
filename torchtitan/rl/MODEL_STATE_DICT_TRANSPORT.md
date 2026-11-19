# Model state dict transport

`ModelStateDictTransport` (`model_state_dict_transport.py`) moves the trainer's
model state dict into generator CPU or GPU memory with
one-sided RDMA reads over a `torch.distributed._transport` backend
(`backend="nixl"`, the default, or `"mooncake"`). It is the default
`Controller.Config.model_state_dict_transport`; select Mooncake with:

```python
config.model_state_dict_transport = ModelStateDictTransport.Config(backend="mooncake")
```

This preserves full synchronization. Set `sync_mode="delta"` to transfer only
changed ranges after the first pull.

It is a separate path rather than a TorchStore transport: the trainer and
generator shardings are known at init, so it plans every read once and stages
straight into registered memory, with no StorageVolume or key lookup. In delta
mode, the trainer keeps the previous staged state in pinned CPU memory. The first
push transfers it in full; later pushes compare the next staged state against
it and transfer only changed tensor ranges. The optimizer can step while
generators read, and the trainer keeps no persistent GPU copy.

`None` keeps TorchStore. TODO: delete the TorchStore path once this transport has
rolled out.

Mooncake needs `mooncake-transfer-engine-cuda13==0.3.14` (installed by
`scripts/install_rl_env.sh`) and a PyTorch nightly from 2026-11-10 or later
(https://github.com/pytorch/pytorch/pull/203623; older ones can drop a
completed read and time out). The vLLM wheel pinned by
`scripts/install_rl_env.sh` pins torch 2026-11-07, which is too old. TODO:
drop this note once that pin moves past 2026-11-10. Without
https://github.com/pytorch/pytorch/pull/204113, host memory on other NUMA nodes
uses node 0's NICs, which can cut read bandwidth several times. On IPv6-only
hosts set `MC_USE_IPV6=1`; Mooncake's handshake otherwise resolves peer
hostnames over IPv4. Without RDMA devices Mooncake falls back to its NVLink
transport, which cannot read pinned CPU memory, and reads fail with
`Mooncake transfer submission failed`; set `MC_FORCE_TCP=1`.

NIXL needs `nixl==1.5.0` and `nixl-cu13==1.5.0` (installed by
`scripts/install_rl_env.sh`) and uses its UCX plugin; select NICs and
transports with the usual UCX variables, e.g. `UCX_NET_DEVICES`, `UCX_TLS` and
`UCX_MAX_RMA_RAILS`.

## Layout

Every generator has the same sharding, so each trainer rank stages one copy of
its shards, laid out per generator rank (g0, g1). A piece is the intersection
of a trainer rank's box of a tensor with a generator rank's box, cast to the
generator's dtype. Generator ranks read their pieces from every trainer rank
straight into their registered prefetch buffers.

Example: one tensor of shape (8, 6), trainer FSDP 4 on dim 0, generator TP 2
on dim 1.

```
        trainer staging buffers (pinned, registered)
        rank 0        rank 1        rank 2        rank 3
       +------+------+------+------+------+------+------+------+
       | g0   | g1   | g0   | g1   | g0   | g1   | g0   | g1   |
       | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  |
       +------+------+------+------+------+------+------+------+
          |      |      |      |      |      |      |      |
          v      |      v      |      v      |      v      |   read
       generator rank 0 (g0): (8, 3), rows 0-1, 2-3, 4-5, 6-7
                 v             v             v             v
       generator rank 1 (g1): (8, 3), rows 0-1, 2-3, 4-5, 6-7
```

Each piece is one read, into the generator rank's prefetch buffer of that key
at consecutive offsets in piece offset order. If every piece of a key is one
contiguous run of the generator shard, as above, that is where it belongs.
Otherwise the key's prefetch buffer holds its pieces packed, and the load
copies them into place: e.g. a row-parallel weight (FSDP `Shard(0)`, TP
`Shard(1)`) read by generator ranks split on dim 0 arrives in pieces that span
only part of each row. This keeps one read per piece, with no per-row reads
and no trainer collective, at the cost of a strided copy on load. When several
trainer ranks stage the same piece (HSDP replicas),
generator ranks are spread across them round-robin. Init raises if the trainer
does not cover every element of a generator shard, or on `Partial` and uneven
`_StridedShard` placements.

## Lifecycle

1. Init (`ActorSystem.initialize_model_state_dict_transport`): every generator
   reports its ranks' `shard_layout`s, which must match
   (`shared_rank_layouts`); each trainer rank allocates and registers one staging
   tensor for full sync, or a baseline and a current-state staging tensor for
   delta sync (`register_trainer_source`); every CPU-prefetch generator rank
   registers its persistent prefetch buffers. A GPU-fetch rank registers two GPU
   staging buffers, each the size of its largest FSDP load group, on its first load.
2. Push: full sync overwrites its staging tensor. For delta sync, the first push
   fills the baseline. Later pushes fill the current-state tensor, then compare its
   packed-piece ranges with the baseline after the device-to-host copies finish.
   Changed ranges are copied into the baseline for the next push. Staging has no
   collectives. The trainer push endpoint returns the changed ranges as ordinary
   control-plane metadata.
3. Pull: CPU prefetch reads every piece for full sync, or only changed pieces for
   delta sync. GPU fetch uses the changed ranges to skip unchanged FSDP groups and
   reads each selected group into a staging buffer while the previous group is
   copied out of the other buffer into its GPU shards (unpacking packed pieces)
   and goes through the existing FSDP all-gather and quantization. Each generator rank raises after `read_timeout_s`
   (600 s).
4. Load: CPU prefetch copies weights from its persistent buffers. GPU fetch uses
   the existing FSDP all-gather and quantization after each group's copy.

The controller remains the only weight-sync coordinator and policy-version
authority. It forwards the changed ranges without interpreting them and finishes
pull N before push N+1, so the trainer can reuse one current-state tensor and
generators can retain unchanged ranges from their previous prefetch state.
Delta transfer does not add a recovery path: transfer or load failures restart
the job, as for full transfer.
