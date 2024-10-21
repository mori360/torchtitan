# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig


# consider split between PP and non-PP
def build_optimizers(model_parts, job_config: JobConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])


def linear_warmup_linear_decay(
    warmup_steps: int, decay_steps: int, current_step: int
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def build_lr_schedulers(optimizers, job_config: JobConfig):
    def _build_lr_scheduler(optimizer):
        """Build a linear warmup and linear decay scheduler"""
        warmup_steps = int(job_config.training.warmup_steps)
        decay_steps = float(max(1, job_config.training.steps - warmup_steps))
        lr_lambda = functools.partial(
            linear_warmup_linear_decay, warmup_steps, decay_steps
        )
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer(
        [_build_lr_scheduler(optimizer) for optimizer in optimizers]
    )


def build_optimizers_in_backward(model_parts, job_config: JobConfig):
    """Wrap one optimizer per param per model part, hooks registered to have .step()
    and .zero_grad() during .backward().
    """

    def _build_optimizer(model):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            # optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            optim_dict = {
                param: torch.optim.Adam([param], **optimizer_kwargs)
                for param in model.parameters()
            }
        elif name == "AdamW":
            raise NotImplementedError(f"Optimizer {name} not supported.")
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for param in model.parameters():
            param.register_post_accumulate_grad_hook(optim_hook)

        optim_ckpt_wrapper = {
            name: optim_dict[param] for name, param in model.named_parameters()
        }
        return optim_ckpt_wrapper

    class OptimizerInBackwardWrapper:
        def __init__(self, optimizers: list[Dict[str, torch.optim.Optimizer]]):
            optims = []
            for optims_from_model in optimizers:
                optims.append(list(optims_from_model.values()))
            self.optimizers = optims

        def state_dict(self) -> list[Dict[str, Any]]:
            """
            Returns a state dict mapping parameter names to optimizer states. This
            state_dict is only loadable by this same class.

            Returns:
                Dict[str, Any]: state dict mapping parameter names to optimizer states.
            """
            state_dicts = []
            for optim_dict in self.optim_map:
                state_dicts.append(
                    {param: opt.state_dict() for param, opt in optim_dict.items()}
                )
            return state_dicts

    return OptimizerInBackwardWrapper(
        [_build_optimizer(model) for model in model_parts]
    )


def build_lr_schedulers_in_backward(optimizers, job_config: JobConfig):
    def _build_lr_scheduler(optimizer):
        """Build a linear warmup and linear decay scheduler"""
        warmup_steps = int(job_config.training.warmup_steps)
        decay_steps = float(max(1, job_config.training.steps - warmup_steps))
        lr_lambda = functools.partial(
            linear_warmup_linear_decay, warmup_steps, decay_steps
        )
        warmup_scheduler = []
        for optim, lr in zip(optimizer, lr_lambda):
            warmup_scheduler.append(LambdaLR(optim, lr_lambda=lr))
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer(
        [_build_lr_scheduler(optimizer) for optimizer in optimizers]
    )
