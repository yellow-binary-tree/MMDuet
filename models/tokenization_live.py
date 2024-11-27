import torch
from transformers import AutoTokenizer, Qwen2Tokenizer
from functools import partial
from .configuration_live import LiveConfigMixin, VideoHeadLiveConfigMixin


def get_stream_placeholder_len(num_frames: int, model_config: VideoHeadLiveConfigMixin) -> str:
    return num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder)


def get_stream_placeholder_jinja2(model_config: VideoHeadLiveConfigMixin) -> str:
    # <image> * (frame_num_tokens * num_frames)
    return f"''.join([{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"


def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin, is_grounding_task) -> torch.Tensor:
    '''
    the start/end idx of every frame_token_interval or stream_end_token after each frame
    '''
    len_frame_placeholder_with_interval = model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.frame_token_interval)
    len_learn = torch.LongTensor([len(model_config.frame_token_interval)] * (num_frames - 1) + [len(model_config.frame_token_interval) if is_grounding_task else len(model_config.stream_end_token)])
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges


def chat_template_llava(self, stream_placeholder):
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + 'system\n' + messages[0]['content'] + eos_token}}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for i in range(messages | length) %}"
            "{% set message = messages[i] %}"
            "{% if message['role'] == 'user' %}"
                "{% if add_stream_query_prompt %}"
                    "{{ eos_token + '\n' + bos_token + 'user\n' + message['content'] + eos_token }}"
                "{% else %}"
                    "{{ '\n' + bos_token + 'user\n' + message['content'] + eos_token }}"
                "{% endif %}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '\n' + bos_token + 'assistant\n' + message['content'] + eos_token }}"
            "{% elif message['role'] == 'stream' and message['num_frames'] > 0 %}"
                "{{ '\n' + bos_token + 'stream\n' + STREAM_PLACEHOLDER + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\n' + bos_token + 'assistant\n' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n' + bos_token + 'stream\n' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ eos_token + '\n' + bos_token + 'assistant\n' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder)
    return template


def chat_template_offsets_llava(tokenizer):
    # now the turn of all roles start with similar beginnings
    def chat_template_transition():
        return {
            (None, 'system'): f'{tokenizer.bos_token}system\n',
            ('system', 'user'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}user\n',
            ('system', 'stream'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}stream\n',
            ('user', 'assistant'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n',
            ('user', 'stream'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}stream\n',
            ('user', 'user'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}user\n',
            ('assistant', 'user'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}user\n',
            ('assistant', 'stream'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}stream\n',
            ('stream', 'user'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}user\n',
            ('stream', 'assistant'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n',
            ('stream', 'stream'): f'{tokenizer.eos_token}\n{tokenizer.bos_token}stream\n',
            'assistant': f'{tokenizer.bos_token}assistant\n',
            'eos_token': tokenizer.eos_token,
        }
    return {k:len(v) for k, v in chat_template_transition().items()}


CHAT_TEMPLATES = {
    'llava': chat_template_llava
}

CHAT_TEMPLATE_OFFSETS = {
    'llava': chat_template_offsets_llava
}


def get_learn_ranges(conversation: list[dict], *, chat_template_offsets: dict[tuple, int], model_config: VideoHeadLiveConfigMixin):
    offset = 0
    learn_ranges = []
    last_role = None
    for message_i, message in enumerate(conversation):
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]
        last_role = role
        if role == 'stream':
            # we do not to use lm_loss to learn anything in the stream
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset, offset + len(message['content']) + chat_template_offsets['eos_token']))
            offset += len(message['content'])
    return learn_ranges


def build_live_tokenizer_and_update_config(llm_pretrained: str, model_config: LiveConfigMixin) -> AutoTokenizer:
    if 'llava' in llm_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(llm_pretrained, use_fast=True, padding_side='left')
        tokenizer.add_special_tokens({'additional_special_tokens': [model_config.v_placeholder]})
        v_placeholder_id = tokenizer.convert_tokens_to_ids(model_config.v_placeholder)
        tokenizer.bos_token, tokenizer.eos_token = "<|im_start|>", "<|im_end|>"

        model_config.update(dict(
            v_placeholder_id=v_placeholder_id,
            eos_token_id=tokenizer.eos_token_id))

        tokenizer.chat_template = CHAT_TEMPLATES['llava'](
            tokenizer,
            get_stream_placeholder_jinja2(model_config),
        )
        tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=CHAT_TEMPLATE_OFFSETS['llava'](tokenizer), model_config=model_config)
        return tokenizer

    else:
        raise NotImplementedError


if __name__ == '__main__':
    chat = [
        {'role': 'system', 'content': 'System message 1.'},
        {'role': 'stream', 'num_frames': 2, 'learn': 1},
        {'role': 'user', 'content': 'User message 1?'},
        {'role': 'assistant', 'content': 'Assistant message 1.', 'learn': True},
        {'role': 'stream', 'num_frames': 3, 'learn': 3},
        {'role': 'assistant', 'content': 'Assistant message 2.', 'learn': True},
        {'role': 'user', 'content': 'User message 2?'},
        {'role': 'stream', 'num_frames': 4, 'learn': 4},
        {'role': 'assistant', 'content': 'Assistant message 3.', 'learn': True},
    ]

    llava_config = VideoHeadLiveConfigMixin(v_placeholder='<image>',
                                   frame_token_cls=True, frame_token_pooled=[3,3], frame_num_tokens=10)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)

    for model_name, tokenizer in [('llava', llava_tokenizer)]:
        print('model name:', model_name)
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        learn_ranges = tokenizer.get_learn_ranges(chat)
        batch = tokenizer([prompt], return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
        print('prompt:', prompt)
        print('batch:', batch)
        print('learn_ranges:', learn_ranges)
        print('learend text:')
        for learn_r in learn_ranges:
            print(prompt[learn_r.start:learn_r.stop], end='\n----------\n')

        batch_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
        for text, labels, input_ids, offset_mapping, learn_range in zip(
            [prompt], batch_labels, batch.input_ids, batch.offset_mapping, [learn_ranges]
        ):
            for learn_r in learn_range:
                start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
                if offset_mapping[:,0][-1] >= learn_r.stop:
                    stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
                else: # the last eos token
                    stop = len(input_ids)
                labels[start-1:stop-1] = input_ids[start:stop]
                labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
        print(batch.input_ids)
        print(batch_labels)