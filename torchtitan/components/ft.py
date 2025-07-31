# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    Callable,
    cast,
    ContextManager,
    Optional,
    TYPE_CHECKING,
    TypeAlias,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.distributed_c10d import ReduceOp
from torchtitan.config.job_config import FaultTolerance as FTConfig
from torchtitan.distributed.pipeline import generate_llm_fqn_per_model_part
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.protocols.train_spec import BaseModelArgs, TrainSpec

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    if TYPE_CHECKING:
        from torchft import local_sgd

    has_torchft = True
else:
    has_torchft = False

FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]


@dataclass
class FaultTolerantTrainSpec(TrainSpec):
    fragment_fn: FragmentFunction | None = None


def module_split(
    model: nn.Module,
    module_names_per_fragment: list[list[str]],
) -> list[nn.Module]:
    """
    This API creates fragments based on specified module names for each fragment.
    This method updates the model in place.

    Args:
        model: The complete model to be split
        module_names_per_fragment: List of lists, where each inner list contains the module names
                               that should be included in that fragment. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "output" for the output projection layer

    Returns:
        List of model fragments

    Example usage:
        module_names_per_fragment = [
            ["tok_embeddings", "layers.0"],     # fragment 0: embeddings + first layer
            ["layers.1", "layers.2"],           # fragment 1: middle layers
            ["norm", "output"]                  # fragment 2: final norm + output
        ]
    """

    def _build_fragment_from_modules(
        fragment_idx: int, module_names: list[str]
    ) -> nn.Module:
        fragment_model = nn.Module()
        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        print(f"fragment {fragment_idx}: Modules to keep: {modules_to_keep}")
        for module_name, module_value in model.named_children():
            # Handle layer-like structures (e.g., "layers.0", "layers.1")
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }

                if not layers_to_keep:
                    continue

                # Keep only specified layers
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name in layers_to_keep:
                            setattr(
                                fragment_model,
                                f"{module_name}.{layer_name}",
                                module_value[layer_name],
                            )
                else:
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = nn.ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(fragment_model, module_name, new_layers)

                continue

            # Handle simple module attributes (e.g., "linear", "norm")
            if module_name not in modules_to_keep:
                continue

            setattr(fragment_model, module_name, module_value)

        return fragment_model

    num_fragments = len(module_names_per_fragment)
    model_fragments = []

    for fragment_idx in range(num_fragments):
        module_names = module_names_per_fragment[fragment_idx]
        model_fragment = _build_fragment_from_modules(
            fragment_idx,
            module_names,
        )
        print(f"building fragment_idx {fragment_idx} " f"with modules {module_names}")
        model_fragments.append(model_fragment)

    return model_fragments


def fragment_llm(
    model: nn.Module,
    ft_config: FTConfig,
    model_config: TransformerModelArgs,
) -> list[nn.Module]:
    assert ft_config.num_fragments > 0

    module_names_per_fragment = ft_config.module_names_per_model_fragment

    input_weight = 1  # Weight for tok_embeddings
    output_weight = 1  # Weight for norm + output layers

    if module_names_per_fragment == []:
        if ft_config.num_fragments == 1:
            return [model]

        module_names_per_fragment = generate_llm_fqn_per_model_part(
            ft_config.num_fragments, model_config.n_layers, input_weight, output_weight
        )

    model_fragments = module_split(model, module_names_per_fragment)
    print(f"Created {len(model_fragments)} model fragments")

    return model_fragments


class FTManager:
    def __init__(
        self,
        ft_config: FTConfig,
    ) -> None:
        if not ft_config.enable:
            self._manager = None
            return

        if not has_torchft:
            raise ImportError("torchft is not installed. Please install it.")

        process_group_timeout = timedelta(
            milliseconds=ft_config.process_group_timeout_ms
        )
        if ft_config.process_group == "gloo":
            pg = ft.ProcessGroupGloo(timeout=process_group_timeout)
        elif ft_config.process_group == "nccl":
            pg = ft.ProcessGroupNCCL(timeout=process_group_timeout)
        else:
            raise ValueError(f"Unsuported process group: {ft_config.process_group}")

        # If the training method is specific, then the quorum should be synchronous
        self.use_async_quorum = ft_config.semi_sync_method is None

        self._manager = ft.Manager(
            pg=pg,
            min_replica_size=ft_config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=self.use_async_quorum,
            replica_id=f"torchtitan_ft_{ft_config.replica_id}",
        )
        self.group_size = ft_config.group_size
        self.replica_id = ft_config.replica_id

        if self.use_async_quorum:
            self.replicate_pg = ft.process_group.ManagedProcessGroup(self._manager)
            self.replicate_pg.register("dp_replicate")

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "ft.Manager":
        assert self._manager is not None
        return self._manager

    def get_dp_info(self, dp_degree: int, dp_rank: int) -> tuple[int, int]:
        if self.enabled:
            return dp_degree * self.group_size, dp_degree * self.replica_id + dp_rank
        else:
            return dp_degree, dp_rank

    def maybe_set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        if self.enabled and self.use_async_quorum:

            def all_reduce_hook(output):
                dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

            def apply_set_all_reduce_hook(m):
                if isinstance(m, FSDPModule):
                    m.set_all_reduce_hook(all_reduce_hook)

            for model_part in model_parts:
                model_part.apply(apply_set_all_reduce_hook)

    @property
    def loss_sync_pg(
        self,
    ) -> Optional["ft.process_group.ManagedProcessGroup"]:
        if self.enabled and self.use_async_quorum:
            return self.replicate_pg
        else:
            # skip loss sync when using semi-sync training
            return None


def maybe_semi_sync_training(
    ft_config: FTConfig,
    ft_manager: FTManager,
    model: torch.nn.Module,
    model_args: BaseModelArgs,
    optimizer: torch.optim.Optimizer,
    train_spec: TrainSpec,
) -> ContextManager[Union["local_sgd.DiLoCo", "local_sgd.LocalSGD", None]]:
    """
    If TorchFT is enabled and the config is set, use semi_sync_method
    """
    semi_sync_method = ft_config.semi_sync_method
    if ft_config.enable and semi_sync_method is not None:
        from torchft import local_sgd

        assert (
            ft_manager._manager is not None
        ), "FTManager must be enabled to use semi-sync training."
        if semi_sync_method.lower() == "diloco":
            train_spec = cast(FaultTolerantTrainSpec, train_spec)
            if train_spec.fragment_fn:
                model_parts = train_spec.fragment_fn(model, ft_config, model_args)
            else:
                model_parts = [model]

            # Create the outer optimizer based on the inner optimizer parameters.
            outer_optimizers = []
            for model in model_parts:
                params = [p for p in model.parameters() if p.requires_grad]
                outer_optimizer = torch.optim.SGD(
                    params, lr=0.7, momentum=0.9, nesterov=True
                )
                outer_optimizers.append(outer_optimizer)

            return local_sgd.DiLoCo(
                manager=ft_manager._manager,
                model_fragments=model_parts,
                inner_optimizer=optimizer,
                outer_optimizer=outer_optimizers,
                sync_every=ft_config.sync_steps,
                should_quantize=ft_config.should_quantize,
                fragment_sync_delay=ft_config.fragment_sync_delay,
                fragment_update_alpha=ft_config.fragment_update_alpha,
            )
        elif semi_sync_method.lower() == "local_sgd":
            assert len(model) == 1
            return local_sgd.LocalSGD(
                manager=ft_manager._manager,
                model=model,
                optimizer=optimizer,
                sync_every=ft_config.sync_steps,
            )
        else:
            raise ValueError(
                f"Unknown training method: {semi_sync_method}, only 'diloco' and 'local_sgd' are supported."
            )
    return nullcontext()
