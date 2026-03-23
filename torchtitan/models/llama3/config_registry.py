# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.components.quantization.qat import QATConverter
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry


def llama3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
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
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def llama3_debugmodel_flex_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel_flex_attn")
    return config


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel_varlen_attn")
    return config


def llama3_debugmodel_opt_in_bwd() -> Trainer.Config:
    config = llama3_debugmodel()
    config.optimizer = OptimizersInBackwardContainer.Config(lr=8e-4)
    return config


def llama3_debugmodel_float8() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
            ),
        ],
    )
    return config


def llama3_debugmodel_float8_emulate() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            Float8LinearConverter.Config(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                emulate=True,
            ),
        ],
    )
    return config


def llama3_debugmodel_lora() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
            ),
        ],
    )
    # For LoRA finetuning, set initial_load_in_hf=True to enable proper
    # checkpoint resumption (load base model from HF, then load LoRA adapters)
    config.checkpoint = CheckpointManager.Config(
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_model_only=False,
    )
    return config


def llama3_debugmodel_qat() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(),
        ],
    )
    return config


def llama3_debugmodel_qat_lora() -> Trainer.Config:
    config = llama3_debugmodel()
    # QATConverter must come before LoRAConverter. See LoRAConverter.convert()
    # for why this ordering is required.
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme="intx_weight_only", group_size=8),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
            ),
        ],
    )
    return config



def llama3_debugmodel_qat_lora_int8act() -> Trainer.Config:
    """QAT (int8_dynamic_act_intx_weight, group_size=8) + LoRA rank=8."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme="int8_dynamic_act_intx_weight", group_size=8),
            LoRAConverter.Config(rank=8, alpha=16.0),
        ],
    )
    return config


def llama3_debugmodel_qat_lora_float8() -> Trainer.Config:
    """QAT (float8_dynamic_act_float8_weight) + LoRA rank=8."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme="float8_dynamic_act_float8_weight"),
            LoRAConverter.Config(rank=8, alpha=16.0),
        ],
    )
    return config


def llama3_debugmodel_qat_lora_merged() -> Trainer.Config:
    """QAT + LoRA with merged save and clean conversion.

    At end of training (last_save_model_only):
      1. LoRA merge: adapters folded into base weights
      2. QAT convert_at_end: FakeQuantizedLinear stripped → plain nn.Linear
    Result: clean checkpoint with plain nn.Linear and QAT-trained weights."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                convert_at_end=True,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=5,
        last_save_model_only=True,
        last_save_in_hf=True,
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=20,
    )
    return config


# ── Draft configs for QAT gap evaluation (not for landing) ──────────────


def llama3_debugmodel_lora_merged() -> Trainer.Config:
    """LoRA fine-tuning with merged save for QAT gap evaluation."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=100,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=100,
        last_save_model_only=True,
    )
    return config


def llama3_debugmodel_qat_lora_gap() -> Trainer.Config:
    """QAT + LoRA with merged + converted save for QAT gap evaluation."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                convert_at_end=True,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=100,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=100,
        last_save_model_only=True,
    )
    return config


def llama3_debugmodel_int4_eval() -> Trainer.Config:
    """Load a checkpoint with intx fake quant applied, run 1 step to measure loss.
    Use --checkpoint.initial_load_path to point to the merged checkpoint."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                convert_at_end=False,  # PREPARE only = fake quant during the 1 step
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=1,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        load_only=True,
        initial_load_model_only=True,
    )
    return config


def llama3_debugmodel_plain_eval() -> Trainer.Config:
    """Load a checkpoint and run 1 step to measure loss (no converters).
    Use --checkpoint.initial_load_path to point to the checkpoint."""
    config = llama3_debugmodel()
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=1,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        load_only=True,
        initial_load_model_only=True,
    )
    return config


# ── QAT gap experiment v2: save fake-quant, convert at eval (not for landing) ──


def llama3_debugmodel_lora_train() -> Trainer.Config:
    """Exp1 Phase A: LoRA-only training for 100 steps.
    Saves full model + LoRA adapters (no merge, no convert)."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=200,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=200,
        last_save_model_only=False,
    )
    config.metrics = MetricsProcessor.Config(log_freq=10)
    return config


def llama3_debugmodel_qat_lora_train() -> Trainer.Config:
    """Exp2 Phase A: QAT PREPARE + LoRA training for 100 steps.
    Saves in fake-quantized state (regular tensors, DCP-compatible)."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme="intx_weight_only", group_size=8),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=200,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=200,
        last_save_model_only=False,
    )
    config.metrics = MetricsProcessor.Config(log_freq=10)
    return config


def llama3_debugmodel_lora_eval() -> Trainer.Config:
    """Exp1 Phase B: Load LoRA checkpoint → merge LoRA → direct PTQ → eval 1 step.
    prepare=False keeps convert() as no-op so model structure matches the
    LoRA-only training checkpoint. finalize() applies direct PTQ (not QAT CONVERT).
    Finalize order (reversed): LoRA merge first, then PTQ.
    Must use NGPU=1 (no FSDP) since real quantized tensors are incompatible
    with DTensor, and LoRA merge needs clean class hierarchy."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                prepare=False,
                convert_at_end=True,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


def llama3_debugmodel_qat_lora_eval() -> Trainer.Config:
    """Exp2 Phase B: Load QAT+LoRA checkpoint → merge LoRA → QAT CONVERT → eval 1 step.
    Finalize order (reversed): LoRA merge first, then QAT CONVERT replaces
    FakeQuantizedLinear with real quantized modules.
    Must use NGPU=1 (no FSDP) since real quantized tensors are incompatible
    with DTensor, and LoRA merge needs clean class hierarchy."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                convert_at_end=True,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# ── Fake-quant eval configs (avoid mslk dependency) ──────────────────
# These use PREPARE (fake quantization) at eval time instead of CONVERT
# (real quantized tensors). The eval loss under fake quant is a close
# proxy for actual quantized inference.


def _make_fq_lora_eval(scheme: str, group_size: int) -> Trainer.Config:
    """Eval for LoRA-only model with fake quantization applied AFTER load+merge.
    Model structure at convert() time matches LoRA-only training (plain Linear
    base + LoRA adapters), so init_weights() produces matching base weights.
    After checkpoint load (adapters only) and LoRA merge, finalize applies
    fake quantization via hooks (DTensor-safe). Works with NGPU=8."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme=scheme,
                group_size=group_size,
                prepare_at_end=True,  # Apply fake quant after load+merge
            ),
            LoRAConverter.Config(rank=8, alpha=16.0, merge_adapter=True),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


def _make_fq_qat_lora_eval(scheme: str, group_size: int) -> Trainer.Config:
    """Eval for QAT+LoRA model with fake quantization (no CONVERT).
    prepare=True so model matches QAT training checkpoint structure.
    LoRA merge happens at finalize. No convert_at_end."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme=scheme,
                group_size=group_size,
                prepare=True,
                convert_at_end=False,  # No real conversion
            ),
            LoRAConverter.Config(rank=8, alpha=16.0, merge_adapter=True),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# LoRA-only eval WITHOUT quantization (merged LoRA, full precision)
def llama3_debugmodel_lora_eval_noquant() -> Trainer.Config:
    """Load LoRA checkpoint → merge LoRA → eval 1 step (no quantization)."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(rank=8, alpha=16.0, merge_adapter=True),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# LoRA-only eval WITH fake quant applied after load
# (PREPARE before load - FakeQuantizedLinear still has weight key so DCP should match)
def llama3_debugmodel_fq_lora_eval_intx() -> Trainer.Config:
    return _make_fq_lora_eval("intx_weight_only", 8)


def llama3_debugmodel_fq_lora_eval_int8act() -> Trainer.Config:
    return _make_fq_lora_eval("int8_dynamic_act_intx_weight", 8)


# QAT+LoRA eval (fake quant preserved, no CONVERT)
def llama3_debugmodel_fq_qat_lora_eval_intx() -> Trainer.Config:
    return _make_fq_qat_lora_eval("intx_weight_only", 8)


def llama3_debugmodel_fq_qat_lora_eval_int8act() -> Trainer.Config:
    return _make_fq_qat_lora_eval("int8_dynamic_act_intx_weight", 8)


# ── Parameterized QAT scheme exploration configs (not for landing) ────


def _make_qat_lora_train(scheme: str, group_size: int) -> Trainer.Config:
    """Factory for QAT+LoRA training configs with different schemes."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme=scheme, group_size=group_size),
            LoRAConverter.Config(rank=8, alpha=16.0),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=200,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=200,
        last_save_model_only=False,
    )
    config.metrics = MetricsProcessor.Config(log_freq=10)
    return config


def _make_lora_eval(scheme: str, group_size: int, prepare: bool) -> Trainer.Config:
    """Factory for eval configs. prepare=True for QAT CONVERT, False for direct PTQ."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme=scheme,
                group_size=group_size,
                prepare=prepare,
                convert_at_end=True,
            ),
            LoRAConverter.Config(rank=8, alpha=16.0, merge_adapter=True),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# int4_weight_only (group_size=128)
def llama3_debugmodel_qat_lora_train_int4() -> Trainer.Config:
    return _make_qat_lora_train("int4_weight_only", 128)


def llama3_debugmodel_lora_eval_int4() -> Trainer.Config:
    return _make_lora_eval("int4_weight_only", 128, prepare=False)


def llama3_debugmodel_qat_lora_eval_int4() -> Trainer.Config:
    return _make_lora_eval("int4_weight_only", 128, prepare=True)


# int8_dynamic_act_intx_weight (group_size=8)
def llama3_debugmodel_qat_lora_train_int8act() -> Trainer.Config:
    return _make_qat_lora_train("int8_dynamic_act_intx_weight", 8)


def llama3_debugmodel_lora_eval_int8act() -> Trainer.Config:
    return _make_lora_eval("int8_dynamic_act_intx_weight", 8, prepare=False)


def llama3_debugmodel_qat_lora_eval_int8act() -> Trainer.Config:
    return _make_lora_eval("int8_dynamic_act_intx_weight", 8, prepare=True)


# float8_dynamic_act_float8_weight (no group_size)
def llama3_debugmodel_qat_lora_train_fp8() -> Trainer.Config:
    return _make_qat_lora_train("float8_dynamic_act_float8_weight", 128)


def llama3_debugmodel_lora_eval_fp8() -> Trainer.Config:
    return _make_lora_eval("float8_dynamic_act_float8_weight", 128, prepare=False)


def llama3_debugmodel_qat_lora_eval_fp8() -> Trainer.Config:
    return _make_lora_eval("float8_dynamic_act_float8_weight", 128, prepare=True)


def llama3_debugmodel_qat_lora_eval_noquant() -> Trainer.Config:
    """Option 1: Load QAT+LoRA checkpoint → merge LoRA → eval (no quantization).
    Provides a full-precision baseline for comparison. Works with NGPU=8.
    intx quantized tensors are incompatible with DTensor, so we skip
    quantization entirely and just evaluate the merged model."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=8,
                prepare=False,
                convert_at_end=False,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# fp8 fake-quant eval configs (DTensor-safe, works with NGPU=8)
def llama3_debugmodel_fq_lora_eval_fp8() -> Trainer.Config:
    return _make_fq_lora_eval("float8_dynamic_act_float8_weight", 128)


def llama3_debugmodel_fq_qat_lora_eval_fp8() -> Trainer.Config:
    return _make_fq_qat_lora_eval("float8_dynamic_act_float8_weight", 128)


# nvfp4
def llama3_debugmodel_qat_lora_train_nvfp4() -> Trainer.Config:
    return _make_qat_lora_train("nvfp4", 128)


def llama3_debugmodel_fq_lora_eval_nvfp4() -> Trainer.Config:
    return _make_fq_lora_eval("nvfp4", 128)


def llama3_debugmodel_fq_qat_lora_eval_nvfp4() -> Trainer.Config:
    return _make_fq_qat_lora_eval("nvfp4", 128)


# mx
def llama3_debugmodel_qat_lora_train_mx() -> Trainer.Config:
    return _make_qat_lora_train("mx", 128)


def llama3_debugmodel_fq_lora_eval_mx() -> Trainer.Config:
    return _make_fq_lora_eval("mx", 128)


def llama3_debugmodel_fq_qat_lora_eval_mx() -> Trainer.Config:
    return _make_fq_qat_lora_eval("mx", 128)


# float8_dynamic_act_int4_weight
def llama3_debugmodel_qat_lora_train_fp8int4() -> Trainer.Config:
    return _make_qat_lora_train("float8_dynamic_act_int4_weight", 128)


def llama3_debugmodel_fq_lora_eval_fp8int4() -> Trainer.Config:
    return _make_fq_lora_eval("float8_dynamic_act_int4_weight", 128)


def llama3_debugmodel_fq_qat_lora_eval_fp8int4() -> Trainer.Config:
    return _make_fq_qat_lora_eval("float8_dynamic_act_int4_weight", 128)


# int4_weight_only with group_size=8 (debugmodel-compatible)
def llama3_debugmodel_qat_lora_train_int4g8() -> Trainer.Config:
    return _make_qat_lora_train("int4_weight_only", 8)


def llama3_debugmodel_fq_lora_eval_int4g8() -> Trainer.Config:
    return _make_fq_lora_eval("int4_weight_only", 8)


def llama3_debugmodel_fq_qat_lora_eval_int4g8() -> Trainer.Config:
    return _make_fq_qat_lora_eval("int4_weight_only", 8)


def llama3_debugmodel_qat_lora_eval_fp8_dtensor() -> Trainer.Config:
    """Option 2: Load QAT+LoRA checkpoint → merge LoRA → fp8 fake quant → eval.
    Uses prepare_at_end=True: model loads as plain Linear (matching checkpoint),
    LoRA merges first, then float8 fake quantization is applied at finalize.
    Fake quant keeps weights as regular tensors (DTensor-compatible), unlike
    real CONVERT which produces Float8Tensor incompatible with DTensor.
    Works with NGPU=8."""
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="float8_dynamic_act_float8_weight",
                prepare_at_end=True,
            ),
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                merge_adapter=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


def llama3_8b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("8B"),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_8b_lora() -> Trainer.Config:
    config = llama3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=128,
                alpha=32.0,
            ),
        ],
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=500,
        initial_load_in_hf=True,
        initial_load_model_only=True,
        last_save_in_hf=True,
    )
    return config


def llama3_8b_lora_merged() -> Trainer.Config:
    """8B LoRA training with merged HF save for lm-eval."""
    config = llama3_8b()
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


# ── 8B full finetune + QAT experiment configs ────


def llama3_8b_finetune() -> Trainer.Config:
    """Full finetune of Llama 3.1 8B from HF checkpoint."""
    config = llama3_8b()
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=50,
        initial_load_in_hf=True,
        initial_load_model_only=True,
    )
    config.training = TrainingConfig(
        local_batch_size=1,
        seq_len=8192,
        steps=200,
    )
    config.metrics = MetricsProcessor.Config(log_freq=10)
    return config


def llama3_8b_qat_finetune() -> Trainer.Config:
    """QAT full finetune of Llama 3.1 8B from HF checkpoint."""
    config = llama3_8b_finetune()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(scheme="intx_weight_only", group_size=128),
        ],
    )
    return config


def llama3_8b_eval_fp() -> Trainer.Config:
    """Eval 8B checkpoint in full precision (no quantization)."""
    config = llama3_8b()
    config.training = TrainingConfig(
        local_batch_size=1,
        seq_len=8192,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


def llama3_8b_eval_fq_intx() -> Trainer.Config:
    """Eval 8B checkpoint with fake quantization (intx_weight_only)."""
    config = llama3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=128,
                prepare_at_end=True,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=1,
        seq_len=8192,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


def llama3_8b_qat_eval_fq_intx() -> Trainer.Config:
    """Eval QAT 8B checkpoint with fake quantization preserved."""
    config = llama3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            QATConverter.Config(
                scheme="intx_weight_only",
                group_size=128,
                prepare=True,
                convert_at_end=False,
            ),
        ],
    )
    config.training = TrainingConfig(
        local_batch_size=1,
        seq_len=8192,
        steps=201,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        interval=1000,
        load_step=200,
        load_only=True,
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return config


# ── 8B QAT+LoRA training configs for lm-eval ────


def _make_8b_qat_lora_train(
    scheme: str,
    group_size: int,
    rank: int = 128,
    alpha: float = 32.0,
    *,
    apply_to_adapters: str = "all",
) -> Trainer.Config:
    """Factory for 8B QAT+LoRA training with merged HF save."""
    config = llama3_8b()
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


def llama3_8b_qat_lora_int4() -> Trainer.Config:
    """8B QAT+LoRA with Int4WeightOnly scheme."""
    return _make_8b_qat_lora_train("int4_weight_only", 128)


def llama3_8b_qat_lora_int8int4() -> Trainer.Config:
    """8B QAT+LoRA with Int8DynamicActivationInt4Weight scheme."""
    return _make_8b_qat_lora_train("int8_dynamic_act_intx_weight", 128)


def llama3_8b_lora_merged_r8() -> Trainer.Config:
    """8B LoRA rank=8 training with merged HF save for lm-eval."""
    config = llama3_8b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
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


def llama3_8b_qat_lora_int4_r8() -> Trainer.Config:
    """8B QAT+LoRA rank=8 with Int4WeightOnly scheme."""
    return _make_8b_qat_lora_train("int4_weight_only", 128, rank=8, alpha=16.0)


def llama3_8b_qat_lora_int8int4_r8() -> Trainer.Config:
    """8B QAT+LoRA rank=8 with Int8DynamicActivationInt4Weight scheme."""
    return _make_8b_qat_lora_train("int8_dynamic_act_intx_weight", 128, rank=8, alpha=16.0)


def llama3_8b_qat_lora_nvfp4() -> Trainer.Config:
    """8B QAT+LoRA with NVFP4 scheme."""
    return _make_8b_qat_lora_train("nvfp4", 128)


def llama3_8b_qat_lora_fp8int4() -> Trainer.Config:
    """8B QAT+LoRA with Float8DynamicActivationInt4Weight scheme."""
    return _make_8b_qat_lora_train("float8_dynamic_act_int4_weight", 128)


# ── 8B QAT(base-only)+LoRA: QAT on base weights, adapters stay full-precision ──


def llama3_8b_qat_lora_int4_noaq() -> Trainer.Config:
    """8B QAT+LoRA Int4WeightOnly, no adapter QAT."""
    return _make_8b_qat_lora_train("int4_weight_only", 128, apply_to_adapters="none")


def llama3_8b_qat_lora_int8int4_noaq() -> Trainer.Config:
    """8B QAT+LoRA Int8DynamicActivationInt4Weight, no adapter QAT."""
    return _make_8b_qat_lora_train(
        "int8_dynamic_act_intx_weight", 128, apply_to_adapters="none"
    )


# ── 8B QAT(lora_a-only)+LoRA rank=8: QAT on base weights + lora_a only ──


def llama3_8b_qat_lora_int4_r8_lora_a() -> Trainer.Config:
    """8B QAT+LoRA rank=8 Int4WeightOnly, QAT on base + lora_a only."""
    return _make_8b_qat_lora_train(
        "int4_weight_only", 128, rank=8, alpha=16.0, apply_to_adapters="lora_a_only"
    )


def llama3_8b_qat_lora_int8int4_r8_lora_a() -> Trainer.Config:
    """8B QAT+LoRA rank=8 Int8DynamicActivationInt4Weight, QAT on base + lora_a only."""
    return _make_8b_qat_lora_train(
        "int8_dynamic_act_intx_weight",
        128,
        rank=8,
        alpha=16.0,
        apply_to_adapters="lora_a_only",
    )


# Also add rank=8 noaq variants for comparison

def llama3_8b_qat_lora_int4_r8_noaq() -> Trainer.Config:
    """8B QAT+LoRA rank=8 Int4WeightOnly, no adapter QAT."""
    return _make_8b_qat_lora_train(
        "int4_weight_only", 128, rank=8, alpha=16.0, apply_to_adapters="none"
    )


def llama3_8b_qat_lora_int8int4_r8_noaq() -> Trainer.Config:
    """8B QAT+LoRA rank=8 Int8DynamicActivationInt4Weight, no adapter QAT."""
    return _make_8b_qat_lora_train(
        "int8_dynamic_act_intx_weight",
        128,
        rank=8,
        alpha=16.0,
        apply_to_adapters="none",
    )


def llama3_70b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("70B"),
        optimizer=OptimizersContainer.Config(lr=1.5e-4),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=8192,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_405b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        profiling=ProfilingConfig(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_registry("405B"),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                Float8LinearConverter.Config(
                    enable_fsdp_float8_all_gather=True,
                    precompute_float8_dynamic_scale_for_fsdp=True,
                    filter_fqns=["output"],
                ),
            ],
        ),
        optimizer=OptimizersContainer.Config(lr=8e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=8192,
            steps=3000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
            enable_async_tensor_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )
