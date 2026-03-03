from swift.llm import register_template, TemplateMeta
from swift.llm.template.template.qwen import Qwen3VLTemplate
from swift.llm.template.template_inputs import StdTemplateInputs
from typing import List, Callable
from swift.llm.template.utils import Context
from dataclasses import dataclass, field
from swift.llm.template.template.utils import DEFAULT_SYSTEM, ChatmlTemplateMeta
from typing import Any, Dict, List, Literal, Optional
from swift.llm.template.utils import Context, Word, findall


@dataclass
class IMTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


class IMTemplate(Qwen3VLTemplate):

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]


register_template(
    IMTemplateMeta('IMTemplate', template_cls=IMTemplate, default_system=None))

