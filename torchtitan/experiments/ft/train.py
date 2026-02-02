# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import AbstractContextManager
from typing import Iterator

import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.train import main, Trainer
from torchtitan.config import JobConfig


class FTTrainer(Trainer):
    ft_manager: FTManager

    def __init__(self, job_config: JobConfig):
        self.ft_manager = FTManager(self.job_config.fault_tolerance)
        super().__init__(job_config)

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

    def init_distributed(self) -> ParallelDims:
        job_config = self.job_config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = job_config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
            ranks=global_ranks,
        )

        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism

        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def get_dp_info(
        self, batch_degree: int, batch_rank: int
    ) -> tuple[int, int]:
        """Override to return FT-aware DP info."""
        return self.ft_manager.get_dp_info(batch_degree, batch_rank)

    def build_loss_fn(self):
        return self.train_spec.build_loss_fn(
            self.job_config, parallel_dims=self.parallel_dims, ft_manager=self.ft_manager
        )

    def build_optimizers(self):
        return self.train_spec.build_optimizers_fn(
            self.model_parts, self.job_config.optimizer, self.parallel_dims, self.ft_manager
        )

    def build_checkpointer(self, model_args) -> CheckpointManager:
        job_config = self.job_config
        return CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(
                    model_args, job_config.model.hf_assets_path
                )
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

    def get_loss_sync_pg(self):
        """Return the process group for loss synchronization."""
        return self.ft_manager.loss_sync_pg

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        from torchtitan.components.loss import IGNORE_INDEX

        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                # pyrefly: ignore [bad-argument-type]
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            ft_pg = self.get_loss_sync_pg()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # loss = local_loss_sum / global_valid_tokens
            # summing across ranks will have result:
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens = global_loss_sum / global_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    def get_profiler_leaf_folder(self) -> str:
        """Return the leaf folder path for profiler outputs."""
        if not self.ft_manager.enabled:
            return ""
        return f"replica_{self.ft_manager.replica_id}"

    def get_training_context(self) -> AbstractContextManager:
        """Return the FT-specific training context manager."""
        job_config = self.job_config
        return maybe_semi_sync_training(
            job_config.fault_tolerance,
            ft_manager=self.ft_manager,
            model=self.model_parts[0],
            n_layers=(
                self.model_args.n_layers
                if hasattr(self.model_args, "n_layers")
                else 0
            ),
            optimizer=self.optimizers,
            fragment_fn=(
                self.train_spec.fragment_fn
                if hasattr(self.train_spec, "fragment_fn")
                else None
            ),
        )


if __name__ == "__main__":
    main(FTTrainer)
