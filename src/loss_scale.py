from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map
from swift.llm.template.utils import ContextType
from collections import Counter


# 需要同时修改框架swift/llm/template/base.py 加入loss_kwargs
# loss_kwargs = inputs.extra_kwargs.copy()
# if hasattr(inputs, 'channel') and inputs.channel is not None:
#     loss_kwargs['channel'] = inputs.channel
# if hasattr(inputs, 'objects') and inputs.objects is not None:
#     loss_kwargs['objects'] = inputs.objects
# res_context_list, loss_scale_list = self.loss_scale(res_context_list, res_context_types, inputs.messages,
#                                                     **loss_kwargs)

class IMLossScale(LossScale):
    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool, **kwargs):
        channel = kwargs.get('channel', 'default')
        task_weights = {
            "holes": 2.0,
            "yuanjiao": 1.0,
            "duyin": 1.0,
            "default": 1.0
        }
        if context_type in {ContextType.RESPONSE, ContextType.SUFFIX}:
            channel_weight = task_weights.get(channel, 1)
            if channel == 'holes':
                holes = kwargs.get("objects", {}).get("ref", [])
                counts = Counter(holes)
                if counts["圆孔"] >= 8 or len(holes) >= 14:
                    channel_weight += 2.0

                return [context], [channel_weight]
            else:
                return [context], [1.0]  # 默认权重


# 注册到 loss_scale_map
loss_scale_map['IMLossScale'] = IMLossScale