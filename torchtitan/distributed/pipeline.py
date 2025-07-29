# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable

from torch import nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


__all__ = [
    "build_pipeline_schedule",
    "generate_split_points",
    "stage_ids_this_rank",
    "generate_module_names_per_stage",
    "module_split",
]


# TODO: It's unclear if this API is general enough to be used by other models.
# If not, we should move it to a Transformer-specific directory.
def generate_split_points(
    schedule_str: str,
    pp_degree: int,
    num_layers: int,
    num_layers_per_stage: int | None,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[str]:
    """
    Generate a list of split points based on the input configs. In this function,
    the number of effective layers considered is the summation of num_layers,
    input_weight, and output_weight.

    If num_layers_per_virtual_stage is given, we require rigid fit of the
    effective layers (regular layers + weighted input + weighted output)
    onto pipeline stages and ranks, with several assertions. It is the users'
    responsibility to figure out the input weight, output weight, and the
    number of regular layers, so that they can be arranged neatly.

    If num_layers_per_virtual_stage is None, we by default set each pipeline rank
    to have 1 stage if schedule_str is a single-stage schedule, or 2 virtual stages
    if it is a multi-stage schedule, and try to distribute all effective layers
    evenly onto the PP stages. If there are extra layers, we disperse them in
    the starting stages.

    Args:
        schedule_str (str): The string of the schedule name.
        pp_degree (int): The pipeline parallel dimension.
        num_layers (int): The number of layers in the model.
        input_weight (int): The number of layers to consider the input modules in layer calculation.
        output_weight (int): The number of layers to consider the output modules in layer calculation.
        num_layers_per_stage (int): The number of layers per (virtual) pipeline stage.

    Returns:
        list[str]: A list of split point FQNs.
    """

    schedule_class = get_schedule_class(schedule_str)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    num_effective_layers = num_layers + input_weight + output_weight

    if num_layers_per_stage is not None:
        # If num_layers_per_stage is provided, we require a rigid fit of the effective layers
        assert num_effective_layers % pp_degree == 0
        num_layers_per_pipeline_rank = num_effective_layers // pp_degree

        assert num_layers_per_pipeline_rank % num_layers_per_stage == 0
        num_stages_per_rank = num_layers_per_pipeline_rank // num_layers_per_stage

        num_total_virtual_stages = num_stages_per_rank * pp_degree
        num_extra_layers = 0

        if is_single_stage_schedule:
            assert (
                num_stages_per_rank == 1
            ), f"Number of stages per rank ({num_stages_per_rank}) must be 1 for single-stage schedules."
        else:
            assert (
                num_stages_per_rank >= 2
            ), f"Number of stages per rank ({num_stages_per_rank}) must be >= 2 for multi-stage schedules."
    else:
        # In a multi-stage schedule, if num_layers_per_stage is not
        # provided, by default each pipeline rank has 2 virtual stages.
        num_stages_per_rank = 1 if is_single_stage_schedule else 2
        num_total_virtual_stages = pp_degree * num_stages_per_rank

        if num_total_virtual_stages > num_effective_layers:
            raise ValueError(
                "The number of total stages cannot be greater than the number of effective layers."
            )

        num_layers_per_stage = num_effective_layers // num_total_virtual_stages
        num_extra_layers = num_effective_layers % num_total_virtual_stages

    assert num_layers_per_stage >= max(input_weight, output_weight)

    splits = []
    current_layer = 0
    for i in range(num_total_virtual_stages - 1):
        if i == 0:
            current_layer += num_layers_per_stage - input_weight
        else:
            current_layer += num_layers_per_stage
        # extra layers will be dispersed to the first stages
        if num_extra_layers > 0:
            current_layer += 1
            num_extra_layers -= 1
        splits.append("layers." + str(current_layer))

    logger.info(
        "No 'pipeline_parallel_split_points' provided. Here is the auto-generated split, "
        f"which may be sub-optimal: {splits}."
    )
    return splits


def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        job_config (JobConfig): The job configuration.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = job_config.parallelism.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.parallelism.pipeline_parallel_schedule
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    microbatch_size = job_config.parallelism.pipeline_parallel_microbatch_size
    batch_size = job_config.training.local_batch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {job_config.training.local_batch_size} must be divisible by number of microbatches {n_microbatches}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )
    n_microbatches = batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {job_config.parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        schedule._load_csv(pp_schedule_csv)

    return schedule


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1), strict=True)
        )
        return stage_v_pairs[pp_rank]


def generate_module_names_per_stage(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """
    Programmatically generates module names per stage for pipeline parallelism with weighting.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (tok_embeddings) in layer calculation
        output_weight: Weight for output modules (norm + output) in layer calculation

    Returns:
        List of lists containing module names for each stage

    Example:
        generate_module_names_per_stage(2, 3, input_weight=2, output_weight=2)
        treats embeddings as 2 layers and norm+output as 2 layers for distribution
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layers.{i}" for i in range(num_layers)]
        return [["tok_embeddings"] + layer_names + ["norm", "output"]]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Ensure each stage gets at least the weight of input/output modules
    if layers_per_stage < max(input_weight, output_weight):
        raise ValueError(
            f"Layers per stage ({layers_per_stage}) must be >= max(input_weight={input_weight}, output_weight={output_weight})"
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("tok_embeddings")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["norm", "output"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def module_split(
    model: nn.Module,
    module_names_per_stage: list[list[str]],
) -> list[nn.Module]:
    """
    This API creates pipeline stages based on specified module names for each stage.
    This method updates the model in place.

    Args:
        model: The complete model to be split
        module_names_per_stage: List of lists, where each inner list contains the module names
                               that should be included in that stage. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "output" for the output projection layer

    Returns:
        List of model chunks

    Example usage:
        module_names_per_stage = [
            ["tok_embeddings", "layers.0"],     # Stage 0: embeddings + first layer
            ["layers.1", "layers.2"],           # Stage 1: middle layers
            ["norm", "output"]                  # Stage 2: final norm + output
        ]
    """

    def _build_stage_from_modules(stage_idx: int, module_names: list[str]) -> nn.Module:
        stage_model = nn.Module()
        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        print(f"Stage {stage_idx}: Modules to keep: {modules_to_keep}")
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
                                stage_model,
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
                    setattr(stage_model, module_name, new_layers)

                continue

            # Handle simple module attributes (e.g., "linear", "norm")
            if module_name not in modules_to_keep:
                continue

            setattr(stage_model, module_name, module_value)

        return stage_model

    num_stages = len(module_names_per_stage)
    models = []

    for stage_idx in range(num_stages):
        module_names = module_names_per_stage[stage_idx]
        model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
        )
        logger.info(f"building stage_idx {stage_idx} " f"with modules {module_names}")
        models.append(model_chunk)

    return models
