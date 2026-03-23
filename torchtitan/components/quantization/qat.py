# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

# Supported scheme names.
_SUPPORTED_SCHEMES = (
    "int4_weight_only",
    "intx_weight_only",
    "int8_dynamic_act_intx_weight",
    "float8_dynamic_act_float8_weight",
    "float8_dynamic_act_int4_weight",
    "nvfp4",
    "mx",
)

# Schemes that accept a group_size parameter.
_SCHEMES_WITH_GROUP_SIZE = (
    "int4_weight_only",
    "intx_weight_only",
    "int8_dynamic_act_intx_weight",
)


def _build_base_config(scheme: str, group_size: int):
    """Return a torchao PTQ base config for the given scheme name."""
    if scheme == "int4_weight_only":
        from torchao.quantization import Int4WeightOnlyConfig

        return Int4WeightOnlyConfig(group_size=group_size)

    elif scheme == "intx_weight_only":
        import torch
        from torchao.quantization import IntxWeightOnlyConfig
        from torchao.quantization.granularity import PerGroup

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        return IntxWeightOnlyConfig(
            weight_dtype=int4_dtype,
            granularity=PerGroup(group_size),
        )

    elif scheme == "int8_dynamic_act_intx_weight":
        import torch
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig
        from torchao.quantization.granularity import PerGroup

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=int4_dtype,
            weight_granularity=PerGroup(group_size),
        )

    elif scheme == "float8_dynamic_act_float8_weight":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        return Float8DynamicActivationFloat8WeightConfig(
            activation_value_lb=1e-12,
        )

    elif scheme == "float8_dynamic_act_int4_weight":
        from torchao.quantization import Float8DynamicActivationInt4WeightConfig

        return Float8DynamicActivationInt4WeightConfig()

    elif scheme == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        return NVFP4DynamicActivationNVFP4WeightConfig()

    elif scheme == "mx":
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

        return MXDynamicActivationMXWeightConfig()

    else:
        raise ValueError(
            f"Unknown QAT scheme '{scheme}'. Supported: {_SUPPORTED_SCHEMES}"
        )


def _build_adapter_qat_config(scheme: str, group_size: int):
    """Build a QATConfig for adapter linears, bypassing _infer_fake_quantize_configs.

    torchao's _infer_fake_quantize_configs has a bug for Int4WeightOnlyConfig:
    version=2 hardcodes group_size=128, version=1 accesses a missing
    zero_point_domain attribute. For adapter linears where group_size=rank
    (often small, e.g. 8), we construct the fake quantize config directly.
    """
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.api import QATStep

    if scheme == "int4_weight_only":
        from torchao.quantization.qat.fake_quantize_config import (
            Int4WeightFakeQuantizeConfig,
        )

        weight_config = Int4WeightFakeQuantizeConfig(
            group_size=group_size,
            activation_dtype=torch.bfloat16,
        )
        return QATConfig(
            activation_config=None,
            weight_config=weight_config,
            step=QATStep.PREPARE,
        )

    elif scheme == "int8_dynamic_act_intx_weight":
        from torchao.quantization.qat.fake_quantize_config import (
            IntxFakeQuantizeConfig,
        )

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        act_config = IntxFakeQuantizeConfig(
            dtype=torch.int8,
            granularity="per_token",
            is_symmetric=False,
        )
        weight_config = IntxFakeQuantizeConfig(
            dtype=int4_dtype,
            group_size=group_size,
            is_symmetric=True,
        )
        return QATConfig(
            activation_config=act_config,
            weight_config=weight_config,
            step=QATStep.PREPARE,
        )

    elif scheme == "intx_weight_only":
        from torchao.quantization.qat.fake_quantize_config import (
            IntxFakeQuantizeConfig,
        )

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        weight_config = IntxFakeQuantizeConfig(
            dtype=int4_dtype,
            group_size=group_size,
            is_symmetric=False,
        )
        return QATConfig(
            activation_config=None,
            weight_config=weight_config,
            step=QATStep.PREPARE,
        )

    else:
        raise ValueError(
            f"Adapter QAT not supported for scheme '{scheme}'. "
            f"Supported: {_SCHEMES_WITH_GROUP_SIZE}"
        )


def apply_qat_prepare(
    model: nn.Module,
    scheme: str,
    group_size: int,
    *,
    filter_fn=None,
) -> None:
    """Apply QAT PREPARE step: insert fake quantization into Linear modules.

    This is the shared core used by both ``QATConverter.convert()`` (all linears)
    and ``LoRAConverter._apply_adapter_qat()`` (adapter linears only).

    After ``quantize_()``, restores ``_init_mean``/``_init_std`` on replaced
    modules and patches ``init_weights`` onto all new module classes so they
    satisfy the Module protocol.

    Args:
        model: The model (or subtree) to quantize.
        scheme: QAT scheme name (must be in ``_SUPPORTED_SCHEMES``).
        group_size: Group size for per-group schemes.
        filter_fn: Optional ``(module, fqn) -> bool`` passed to ``quantize_()``.
            When None, all ``nn.Linear`` modules are quantized.
    """
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.api import QATStep

    from torchtitan.models.common.linear import Linear

    # Snapshot init params before quantize_ replaces Linear subclasses
    # with FakeQuantizedLinear (which lacks _init_mean / _init_std).
    init_params_cache: dict[str, tuple[float, float]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod, "_init_mean"):
            init_params_cache[name] = (
                getattr(mod, "_init_mean", 0.0),
                getattr(mod, "_init_std", 0.02),
            )

    base_config = _build_base_config(scheme, group_size)

    kwargs = {}
    if filter_fn is not None:
        kwargs["filter_fn"] = filter_fn
    quantize_(model, QATConfig(base_config, step=QATStep.PREPARE), **kwargs)

    # Restore init params on replaced modules and patch init_weights onto
    # all new module classes introduced by quantize_() so they satisfy
    # the Module protocol's init_weights contract.
    _patched_classes: set[type] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in init_params_cache:
            init_mean, init_std = init_params_cache[name]
            object.__setattr__(mod, "_init_mean", init_mean)
            object.__setattr__(mod, "_init_std", init_std)

        cls = type(mod)
        if cls not in _patched_classes and not hasattr(cls, "init_weights"):
            if isinstance(mod, nn.Linear):
                cls.init_weights = Linear.init_weights
            else:
                cls.init_weights = lambda self, **kwargs: None
            _patched_classes.add(cls)


def _apply_fq_hooks(
    model: nn.Module,
    scheme: str,
    group_size: int,
) -> None:
    """Apply fake-quantization via forward pre-hooks (DTensor-safe).

    Unlike ``apply_qat_prepare`` which calls ``quantize_()`` to replace module
    classes, this function registers forward pre-hooks on each ``nn.Linear``
    that fake-quantize the weight in-place before the forward call and restore
    the original weight afterward.  This keeps the module class unchanged so
    FSDP wrappers and DTensor parameters are preserved.
    """
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.api import QATStep
    from torchao.quantization.qat.linear import FakeQuantizedLinear

    base_config = _build_base_config(scheme, group_size)
    qat_config = QATConfig(base_config, step=QATStep.PREPARE)

    # Build a temporary FakeQuantizedLinear to extract the weight fake quantizer
    # config. This avoids duplicating torchao's config inference logic.
    from torchao.quantization.qat.api import _infer_fake_quantize_configs

    act_fq_config, weight_fq_config = _infer_fake_quantize_configs(base_config)

    count = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Skip modules that are already fake-quantized
        if isinstance(mod, FakeQuantizedLinear):
            continue

        # Create a weight fake quantizer for this module
        from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

        if weight_fq_config is not None:
            fq = FakeQuantizerBase.from_config(weight_fq_config)
            fq = fq.to(device=mod.weight.device)

            def _make_hook(fake_quantizer):
                def hook(module, args):
                    # Fake-quantize weight in-place: dequant(quant(w))
                    with torch.no_grad():
                        w = module.weight
                        # Use to_local() to get the plain tensor for fake
                        # quantization, then redistribute back.
                        from torch.distributed._tensor import DTensor

                        if isinstance(w, DTensor):
                            local_w = w.to_local()
                            fq_local = fake_quantizer(local_w)
                            # Replace the local data in-place
                            w._local_tensor.copy_(fq_local)
                        else:
                            fq_w = fake_quantizer(w)
                            module.weight.data.copy_(fq_w)

                return hook

            mod.register_forward_pre_hook(_make_hook(fq))
            count += 1

    logger.info(
        f"Registered fake-quantization hooks on {count} Linear modules "
        f"(scheme={scheme}, group_size={group_size})"
    )


class QATConverter(Configurable):
    """Apply quantization-aware training via torchao's QATConfig.

    Uses ``torchao.quantize_(model, QATConfig(base_config, step="prepare"))``
    to insert fake quantization into ``nn.Linear`` modules. The ``scheme``
    config field selects a torchao PTQ base config, which QATConfig uses to
    infer the appropriate fake quantization for both weights and activations.

    Supported schemes:
      - ``"int4_weight_only"`` — int4 weight-only fake quantization
      - ``"intx_weight_only"`` — intx weight-only fake quantization
      - ``"int8_dynamic_act_intx_weight"`` — int8 activation + int4 weight
      - ``"float8_dynamic_act_float8_weight"`` — float8 activation + float8 weight
      - ``"float8_dynamic_act_int4_weight"`` — float8 activation + int4 weight
      - ``"nvfp4"`` — NVFP4 dynamic activation + NVFP4 weight
      - ``"mx"`` — MX dynamic activation + MX weight

    When composed with LoRA (QATConverter listed before LoRAConverter in converters),
    LoRA will inherit from FakeQuantizedLinear so base weights are fake-quantized
    while LoRA adapters stay full-precision.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        scheme: str = "int4_weight_only"
        """QAT scheme name. Maps to a torchao PTQ base config.
        Supported: 'int4_weight_only', 'intx_weight_only',
        'int8_dynamic_act_intx_weight', 'float8_dynamic_act_float8_weight',
        'float8_dynamic_act_int4_weight', 'nvfp4', 'mx'."""

        group_size: int = 128
        """Group size for per-group weight quantization.
        Used by schemes that support per-group granularity
        (int4_weight_only, intx_weight_only, int8_dynamic_act_intx_weight).
        Must divide in_features of all Linear layers in the model."""

        convert_at_end: bool = False
        """When True, finalize() strips FakeQuantizedLinear back to plain
        nn.Linear (preserving weights). Combined with LoRA merge_adapter=True,
        this produces a clean checkpoint with trained weights and no fake
        quantization wrappers.

        To produce a real quantized model instead, set quantize_at_end=True."""

        quantize_at_end: bool = False
        """When True (and convert_at_end=True), finalize() applies real
        quantization (QAT CONVERT or direct PTQ) instead of just stripping
        FakeQuantizedLinear. The resulting model has packed quantized weights
        (e.g. IntxWeightOnlyQuantizedLinear)."""

        prepare: bool = True
        """When True (default), convert() inserts fake quantization (PREPARE step).
        When False, convert() is a no-op. Use prepare=False for eval configs
        where the training checkpoint was saved without QAT — the model
        structure must match the checkpoint for DCP to load correctly."""

        apply_to_adapters: str = "all"
        """Controls whether QAT fake quantization is applied to LoRA adapter
        linears (in addition to the base model weights which always get QAT).

        - ``"all"``: Apply QAT to both lora_a and lora_b adapter linears.
        - ``"lora_a_only"``: Apply QAT only to lora_a (the down-projection).
          lora_b (up-projection, shape out_features×rank) stays full-precision.
          This matches unsloth's approach of skipping modules with small
          in_features.
        - ``"none"``: No QAT on adapter linears; only base weights get fake
          quantization. Recommended based on experimental evidence showing
          adapter QAT is unnecessary and hurts training efficiency.
        """

        prepare_at_end: bool = False
        """When True, finalize() applies PREPARE (fake quantization) instead of
        CONVERT. Use this for evaluating a non-QAT checkpoint with simulated
        quantization: convert() is a no-op (prepare=False implied), then after
        checkpoint load and LoRA merge, finalize() inserts fake quantization
        and runs 1 eval step."""

    _VALID_ADAPTER_MODES = {"all", "lora_a_only", "none"}

    def __init__(self, config: Config, **kwargs):
        if config.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unknown QAT scheme '{config.scheme}'. "
                f"Supported: {_SUPPORTED_SCHEMES}"
            )
        # Backward compat: accept bool for apply_to_adapters
        if isinstance(config.apply_to_adapters, bool):
            apply_to_adapters = "all" if config.apply_to_adapters else "none"
        else:
            apply_to_adapters = config.apply_to_adapters
        if apply_to_adapters not in self._VALID_ADAPTER_MODES:
            raise ValueError(
                f"Invalid apply_to_adapters='{apply_to_adapters}'. "
                f"Must be one of {self._VALID_ADAPTER_MODES}"
            )
        self.scheme = config.scheme
        self.group_size = config.group_size
        self.convert_at_end = config.convert_at_end
        self.quantize_at_end = config.quantize_at_end
        self.prepare = config.prepare
        self.apply_to_adapters = apply_to_adapters
        self.prepare_at_end = config.prepare_at_end
        if self.prepare_at_end:
            # prepare_at_end implies prepare=False (no PREPARE at convert time)
            self.prepare = False
            self.convert_at_end = True  # needed so finalize() runs
        if config.scheme not in _SCHEMES_WITH_GROUP_SIZE:
            logger.warning(
                f"QAT scheme '{config.scheme}' does not use group_size, "
                f"ignoring group_size={config.group_size}"
            )
        logger.info(
            f"QAT active (scheme={self.scheme}, group_size={self.group_size}, "
            f"prepare={self.prepare}, convert_at_end={self.convert_at_end})"
        )

    def convert(self, model: nn.Module) -> None:
        if not self.prepare:
            return

        apply_qat_prepare(model, self.scheme, self.group_size)

        # Store QAT config on the model so downstream converters (e.g. LoRA)
        # can apply the same QAT to newly created modules.
        if self.apply_to_adapters != "none":
            model._qat_scheme = self.scheme  # type: ignore[attr-defined]
            model._qat_group_size = self.group_size  # type: ignore[attr-defined]
            model._qat_adapter_mode = self.apply_to_adapters  # type: ignore[attr-defined]

        logger.info(
            f"Applied QAT fake quantization (scheme={self.scheme}, "
            f"group_size={self.group_size})"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass

    def finalize(self, model: nn.Module) -> None:
        """Finalize QAT model at end of training.

        Three modes depending on config:

        1. convert_at_end=False (default): no-op.

        2. convert_at_end=True, quantize_at_end=False (default):
           Strip FakeQuantizedLinear → plain nn.Linear (weights preserved).
           Produces a clean checkpoint with QAT-trained weights.

        3. convert_at_end=True, quantize_at_end=True:
           Apply real quantization (FakeQuantizedLinear → quantized modules
           with packed weights, or direct PTQ if prepare=False).

        4. prepare_at_end=True:
           Apply fake quantization hooks for eval (DTensor-safe).

        LoRA adapters must be merged before any conversion. If adapters are
        still present, finalize is skipped with a warning.
        """
        if not self.convert_at_end:
            return

        # prepare_at_end: apply fake quantization hooks for eval.
        if self.prepare_at_end:
            _apply_fq_hooks(model, self.scheme, self.group_size)
            logger.info(
                f"Applied fake quantization hooks at finalize "
                f"(scheme={self.scheme}, group_size={self.group_size})"
            )
            return

        # Check for remaining LoRA adapters — if present, skip.
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_a") or hasattr(mod, "lora_b"):
                logger.warning(
                    "QAT finalize skipped: LoRA adapters still present. "
                    "Use LoRA merge_adapter=True to merge adapters first."
                )
                return

        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep

        if not self.quantize_at_end:
            # Strip FakeQuantizedLinear → nn.Linear (no real quantization).
            # base_config=None tells torchao to just unwrap the fake quant
            # wrapper, preserving the original weights.
            quantize_(model, QATConfig(base_config=None, step=QATStep.CONVERT))
            logger.info(
                "Stripped FakeQuantizedLinear → nn.Linear (weights preserved)"
            )
            return

        # Real quantization path
        base_config = _build_base_config(self.scheme, self.group_size)

        if self.prepare:
            # QAT CONVERT: FakeQuantizedLinear → real quantized modules
            quantize_(model, QATConfig(base_config, step=QATStep.CONVERT))
            method = "QAT CONVERT"
        else:
            # Direct PTQ: plain Linear → real quantized modules
            quantize_(model, base_config)
            method = "direct PTQ"

        logger.info(
            f"Applied {method} (scheme={self.scheme}, group_size={self.group_size})"
        )


