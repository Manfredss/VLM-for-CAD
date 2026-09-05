"""Layer-wise learning-rate optimizer plugin for ms-swift.

The plugin intentionally has no project-local imports so ms-swift can load it
early via ``--external_plugins``.  It supports both:

* ms-swift >= 4.1: ``swift.optimizers`` callback API;
* older ms-swift releases: ``swift.plugin`` function API.

Vision, multimodal-aligner and language-model parameters receive ``vit_lr``,
``aligner_lr`` and ``learning_rate`` respectively.  PEFT/LoRA wrappers are
handled by parameter identity, so a trainable parameter is never registered
twice.  Any trainable parameter not described by the model metadata falls back
to the language-model learning rate instead of being silently dropped.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

from transformers import Trainer as HfTrainer


_SWIFT_API: str | None = None
_IMPORT_ERROR: Exception | None = None

try:  # ms-swift 4.1+
    from swift.optimizers.base import OptimizerCallback
    from swift.optimizers.mapping import optimizers_map
    from swift.optimizers.multimodal import get_param_startswith

    _SWIFT_API = "callback"
except ImportError:  # pragma: no cover - depends on runtime image
    try:  # ms-swift <= 4.0
        from swift.plugin import optimizers_map
        from swift.plugin.optimizer import get_param_startswith

        _SWIFT_API = "function"
    except ImportError as exc_old:  # pragma: no cover - depends on runtime image
        _IMPORT_ERROR = exc_old

try:
    from swift.utils import get_logger

    logger = get_logger()
except ImportError:  # pragma: no cover - useful for local syntax/unit checks
    logger = logging.getLogger(__name__)


def _trainable_named(parameters: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Materialise a helper result and discard frozen tensors."""

    return [(name, param) for name, param in parameters if param.requires_grad]


def _model_arch(model: Any) -> Any | None:
    """Find ms-swift's model architecture metadata through common wrappers."""

    queue = [model]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        meta = getattr(current, "model_meta", None)
        arch = getattr(meta, "model_arch", None)
        if arch is not None:
            return arch
        for attr in ("model", "base_model", "module"):
            child = getattr(current, attr, None)
            if child is not current:
                queue.append(child)
    return None


def _metadata_groups(model: Any) -> tuple[list, list, list] | None:
    """Return (vision, aligner, llm) using ms-swift's architecture metadata."""

    arch = _model_arch(model)
    if arch is None:
        return None
    try:
        # The third argument is an exclusion prefix in both supported APIs.
        vision = get_param_startswith(model, arch.vision_tower, arch.aligner)
        aligner = get_param_startswith(model, arch.aligner)
        llm = get_param_startswith(model, arch.language_model)
        return (
            _trainable_named(vision),
            _trainable_named(aligner),
            _trainable_named(llm),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "Cannot use model_arch for LR grouping; using name fallback: %s", exc
        )
        return None


def _name_fallback_groups(model: Any) -> tuple[list, list, list]:
    """Conservative fallback for a new model architecture not yet known to swift."""

    vision_markers = (
        ".visual.",
        ".vision_tower.",
        ".vision_model.",
        ".image_encoder.",
        ".vit.",
    )
    aligner_markers = (
        ".aligner.",
        ".merger.",
        ".mm_projector.",
        ".multi_modal_projector.",
        ".vision_projection.",
    )
    vision: list[tuple[str, Any]] = []
    aligner: list[tuple[str, Any]] = []
    llm: list[tuple[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        dotted = f".{name.lower()}."
        if any(marker in dotted for marker in aligner_markers):
            aligner.append((name, param))
        elif any(marker in dotted for marker in vision_markers):
            vision.append((name, param))
        else:
            llm.append((name, param))
    return vision, aligner, llm


def _decay_parameter_names(model: Any) -> set[str]:
    """Bridge the bound/unbound Trainer helper signatures across transformers."""

    try:
        return set(HfTrainer.get_decay_parameter_names(None, model))
    except TypeError:  # pragma: no cover - transformers API-dependent
        return set(HfTrainer.get_decay_parameter_names(model))


def _unique_groups(model: Any) -> tuple[list, list, list]:
    """Deduplicate groups and put every trainable parameter in exactly one."""

    proposed = _metadata_groups(model) or _name_fallback_groups(model)
    assigned: set[int] = set()
    result: list[list[tuple[str, Any]]] = [[], [], []]

    # Aligner must win over vision when an architecture reports overlapping
    # prefixes.  The final returned order remains vision, aligner, llm.
    for output_index, source_index in ((1, 1), (0, 0), (2, 2)):
        for name, param in proposed[source_index]:
            if not param.requires_grad or id(param) in assigned:
                continue
            assigned.add(id(param))
            result[output_index].append((name, param))

    # New Qwen architectures occasionally add small trainable modules before
    # swift's model metadata is updated.  Train them at the conservative LLM LR.
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in assigned:
            assigned.add(id(param))
            result[2].append((name, param))

    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned != expected:
        raise RuntimeError(
            "Optimizer grouping invariant failed: "
            f"assigned={len(assigned)}, trainable={len(expected)}"
        )
    return result[0], result[1], result[2]


def _build_grouped_params(args: Any, model: Any) -> list[dict[str, Any]]:
    vision, aligner, llm = _unique_groups(model)
    vit_lr = getattr(args, "vit_lr", None)
    aligner_lr = getattr(args, "aligner_lr", None)

    def env_true(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    # Some PEFT checkpoints already contain ViT/aligner LoRA tensors.  Swift's
    # freeze flags can run before adapter restoration, which makes those loaded
    # LoRA tensors trainable again.  Enforce the requested freeze here, after
    # all adapters have been materialised and immediately before optimizer
    # creation, so a protocol-only replay run cannot drift visual grounding.
    for group_name, freeze, source in (
        (
            "vision",
            bool(getattr(args, "freeze_vit", False))
            or env_true("CAD_FREEZE_VIT")
            or (vit_lr is not None and float(vit_lr) == 0.0),
            vision,
        ),
        (
            "aligner",
            bool(getattr(args, "freeze_aligner", False))
            or env_true("CAD_FREEZE_ALIGNER")
            or (aligner_lr is not None and float(aligner_lr) == 0.0),
            aligner,
        ),
    ):
        if not freeze:
            continue
        tensor_count = len(source)
        scalar_count = sum(param.numel() for _, param in source)
        for _, param in source:
            param.requires_grad_(False)
        source.clear()
        logger.info(
            "optimizer hard-freeze group=%s tensors=%d params=%d",
            group_name,
            tensor_count,
            scalar_count,
        )

    decay_names = _decay_parameter_names(model)

    learning_rates = (
        args.learning_rate if vit_lr is None else vit_lr,
        args.learning_rate if aligner_lr is None else aligner_lr,
        args.learning_rate,
    )
    names = ("vision", "aligner", "llm_or_other")
    sources = (vision, aligner, llm)
    grouped: list[dict[str, Any]] = []

    for group_name, lr, source in zip(names, learning_rates, sources):
        tensor_count = len(source)
        scalar_count = sum(param.numel() for _, param in source)
        logger.info(
            "optimizer group=%s lr=%s trainable_tensors=%d trainable_params=%d",
            group_name,
            lr,
            tensor_count,
            scalar_count,
        )
        for use_decay, weight_decay in (
            (False, 0.0),
            (True, float(args.weight_decay)),
        ):
            params = [
                param for name, param in source if (name in decay_names) == use_decay
            ]
            if params:
                grouped.append(
                    {
                        "params": params,
                        "weight_decay": weight_decay,
                        "lr": lr,
                    }
                )

    if not grouped:
        raise RuntimeError(
            "No trainable parameters were found. Check freeze_vit/freeze_aligner "
            "and LoRA target_modules."
        )
    return grouped


def _make_optimizer(args: Any, model: Any) -> Any:
    grouped = _build_grouped_params(args, model)
    optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(
        args, model
    )
    return optimizer_cls(grouped, **optimizer_kwargs)


if _SWIFT_API == "callback":

    class CADLayerwiseOptimizer(OptimizerCallback):
        """ms-swift 4.1 optimizer callback."""

        def create_optimizer(self):
            return _make_optimizer(self.args, self.trainer.model)

    optimizers_map["CADLayerwiseOptimizer"] = CADLayerwiseOptimizer

elif _SWIFT_API == "function":

    def create_optimizer(args, model, dataset=None):  # noqa: ARG001
        """Legacy ms-swift optimizer factory."""

        return _make_optimizer(args, model), None

    optimizers_map["CADLayerwiseOptimizer"] = create_optimizer

else:  # Make a broken training image fail with an actionable message.
    raise ImportError(
        "Cannot register CADLayerwiseOptimizer: install ms-swift 4.1.3 or a "
        "compatible older swift.plugin release."
    ) from _IMPORT_ERROR
