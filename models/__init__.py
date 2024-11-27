from transformers import HfArgumentParser

from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llava.video_head_live_llava_qwen import build_video_head_live_llava_qwen
from .modeling_live import fast_greedy_generate


def build_model_and_tokenizer(is_training, **kwargs):
    llm_pretrained = kwargs.get('llm_pretrained', None)
    if 'llava' in llm_pretrained:
        return build_video_head_live_llava_qwen(is_training=is_training, **kwargs)
    else:
        raise NotImplementedError(f'Not support {llm_pretrained}')

def parse_args(live_version=None) -> LiveTrainingArguments:
    if live_version is None:
        args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
        live_version = args.live_version
    args, = HfArgumentParser(get_args_class(live_version)).parse_args_into_dataclasses()
    return args