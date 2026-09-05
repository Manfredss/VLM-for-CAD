"""
IMoptimizer — 分层学习率优化器（兼容 ms-swift 4.0.0.dev0 和 3.x）
适配 Qwen3.5-27B
"""
import logging
from transformers import Trainer

try:
    from swift.plugin import optimizers_map
    from swift.plugin.optimizer import get_param_startswith
    _HAS_SWIFT_PLUGIN = True
except ImportError:
    _HAS_SWIFT_PLUGIN = False

try:
    from swift.utils import get_logger
    logger = get_logger()
except ImportError:
    logger = logging.getLogger(__name__)


def create_optimizer(args, model, dataset):
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch

    vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
    aligner_parameters = get_param_startswith(model, model_arch.aligner)
    llm_parameters = get_param_startswith(model, model_arch.language_model)

    optimizer_grouped_parameters = []

    vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate
    aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate

    logger.info(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.learning_rate}')

    for lr, parameters in zip([vit_lr, aligner_lr, args.learning_rate],
                              [vit_parameters, aligner_parameters, llm_parameters]):
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in parameters if n in decay_parameters]
            else:
                params = [p for n, p in parameters if n not in decay_parameters]
            if not params:
                continue
            optimizer_grouped_parameters.append({
                'params': params,
                'weight_decay': wd,
                'lr': lr,
            })

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


if _HAS_SWIFT_PLUGIN:
    optimizers_map['IMoptimizer'] = create_optimizer
