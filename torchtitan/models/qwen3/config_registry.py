# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.qat import QATConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from . import model_registry


def qwen3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_debugmodel_flex() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_flex"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_0_6b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("0.6B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_0_6b_finetune() -> Trainer.Config:
    """Full finetune of Qwen3 0.6B from HF checkpoint."""
    config = qwen3_0_6b()
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=100,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    config.training = TrainingConfig(
        local_batch_size=4,
        seq_len=4096,
        steps=500,
    )
    config.metrics = MetricsProcessor.Config(log_freq=10)
    return config


def qwen3_0_6b_lora_merged() -> Trainer.Config:
    """0.6B LoRA training with merged HF save for lm-eval."""
    config = qwen3_0_6b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=128,
                alpha=32.0,
                merge_adapter=True,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    return config


def _make_0_6b_qat_lora_train(
    scheme: str,
    group_size: int,
    rank: int = 128,
    alpha: float = 32.0,
    *,
    apply_to_adapters: str = "all",
) -> Trainer.Config:
    """Factory for 0.6B QAT+LoRA training with merged HF save."""
    config = qwen3_0_6b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme=scheme,
                group_size=group_size,
                convert_at_end=True,
                quantize_at_end=False,
                apply_to_adapters=apply_to_adapters,
            ),
            LoRAConverter.Config(
                rank=rank,
                alpha=alpha,
                merge_adapter=True,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    return config


def qwen3_0_6b_qat_lora_int4() -> Trainer.Config:
    """0.6B QAT+LoRA with Int4WeightOnly scheme."""
    return _make_0_6b_qat_lora_train("int4_weight_only", 128)


def qwen3_0_6b_qat_lora_int8int4() -> Trainer.Config:
    """0.6B QAT+LoRA with Int8DynamicActivationInt4Weight scheme."""
    return _make_0_6b_qat_lora_train("int8_dynamic_act_intx_weight", 128)


# ── 0.6B QAT(base-only)+LoRA: QAT on base weights, adapters stay full-precision ──


def qwen3_0_6b_qat_lora_int4_noaq() -> Trainer.Config:
    """0.6B QAT+LoRA Int4WeightOnly, no adapter QAT."""
    return _make_0_6b_qat_lora_train("int4_weight_only", 128, apply_to_adapters="none")


def qwen3_0_6b_qat_lora_int8int4_noaq() -> Trainer.Config:
    """0.6B QAT+LoRA Int8DynamicActivationInt4Weight, no adapter QAT."""
    return _make_0_6b_qat_lora_train(
        "int8_dynamic_act_intx_weight", 128, apply_to_adapters="none"
    )


def qwen3_8b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-8B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("8B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=2e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def qwen3_8b_lora_merged() -> Trainer.Config:
    """8B LoRA training with merged HF save for lm-eval."""
    config = qwen3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=128,
                alpha=32.0,
                merge_adapter=True,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    return config


def _make_8b_qat_lora_train(
    scheme: str,
    group_size: int,
    rank: int = 128,
    alpha: float = 32.0,
    *,
    apply_to_adapters: str = "all",
) -> Trainer.Config:
    """Factory for 8B QAT+LoRA training with merged HF save."""
    config = qwen3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme=scheme,
                group_size=group_size,
                convert_at_end=True,
                quantize_at_end=False,
                apply_to_adapters=apply_to_adapters,
            ),
            LoRAConverter.Config(
                rank=rank,
                alpha=alpha,
                merge_adapter=True,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    return config


def qwen3_8b_qat_lora_int4() -> Trainer.Config:
    """8B QAT+LoRA with Int4WeightOnly scheme."""
    return _make_8b_qat_lora_train("int4_weight_only", 128)


def qwen3_8b_qat_lora_int8int4() -> Trainer.Config:
    """8B QAT+LoRA with Int8DynamicActivationInt4Weight scheme."""
    return _make_8b_qat_lora_train("int8_dynamic_act_intx_weight", 128)


# ── 8B QAT(base-only)+LoRA: QAT on base weights, adapters stay full-precision ──


def qwen3_8b_qat_lora_int4_noaq() -> Trainer.Config:
    """8B QAT+LoRA Int4WeightOnly, no adapter QAT."""
    return _make_8b_qat_lora_train("int4_weight_only", 128, apply_to_adapters="none")


def qwen3_8b_qat_lora_int8int4_noaq() -> Trainer.Config:
    """8B QAT+LoRA Int8DynamicActivationInt4Weight, no adapter QAT."""
    return _make_8b_qat_lora_train(
        "int8_dynamic_act_intx_weight", 128, apply_to_adapters="none"
    )


def qwen3_1_7b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_14b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_registry("14B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def qwen3_32b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def qwen3_moe_debug() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_moe"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )
