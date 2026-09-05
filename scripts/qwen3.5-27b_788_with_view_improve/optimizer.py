"""
IMoptimizer — Hierarchical learning rate optimizer for Qwen3.5-27B.

Functionally equivalent to swift.optimizers.multimodal.MultimodalOptimizerCallback;
kept as a separate plugin so it stays a customization point.

Compatible with both ms-swift 4.1.x (class-based, swift.optimizers.*) and
ms-swift 4.0.x (function-based, swift.plugin.*).
"""
import logging
from transformers import Trainer as HfTrainer

_API = None
try:
    from swift.optimizers.mapping import optimizers_map
    from swift.optimizers.multimodal import get_param_startswith
    from swift.optimizers.base import OptimizerCallback
    _API = 'new'
except ImportError:
    try:
        from swift.plugin import optimizers_map
        from swift.plugin.optimizer import get_param_startswith
        _API = 'legacy'
    except ImportError:
        pass

try:
    from swift.utils import get_logger
    logger = get_logger()
except ImportError:
    logger = logging.getLogger(__name__)


def _build_grouped_params(args, model):
    decay_parameters = set(HfTrainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch
    vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
    aligner_parameters = get_param_startswith(model, model_arch.aligner)
    llm_parameters = get_param_startswith(model, model_arch.language_model)

    vit_lr = args.vit_lr if args.vit_lr is not None else args.learning_rate
    aligner_lr = args.aligner_lr if args.aligner_lr is not None else args.learning_rate
    logger.info(f'vit_lr: {vit_lr}, aligner_lr: {aligner_lr}, llm_lr: {args.learning_rate}')

    grouped = []
    for lr, parameters in zip([vit_lr, aligner_lr, args.learning_rate],
                              [vit_parameters, aligner_parameters, llm_parameters]):
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in parameters if n in decay_parameters]
            else:
                params = [p for n, p in parameters if n not in decay_parameters]
            if params:
                grouped.append({'params': params, 'weight_decay': wd, 'lr': lr})
    return grouped


if _API == 'new':
    class IMOptimizerCallback(OptimizerCallback):
        def create_optimizer(self):
            args, model = self.args, self.trainer.model
            grouped = _build_grouped_params(args, model)
            optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
            return optimizer_cls(grouped, **optimizer_kwargs)

    optimizers_map['IMoptimizer'] = IMOptimizerCallback

elif _API == 'legacy':
    def create_optimizer(args, model, dataset):
        grouped = _build_grouped_params(args, model)
        optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args, model)
        return optimizer_cls(grouped, **optimizer_kwargs), None

    optimizers_map['IMoptimizer'] = create_optimizer
