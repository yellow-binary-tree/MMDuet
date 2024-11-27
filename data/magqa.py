import random, json, tqdm
import numpy as np
import torch
from .stream import StreamMixIn
from .utils import ceil_time_by_fps, floor_time_by_fps, rand_bool, DictWithTo, reformat_example_for_debug

from transformers.utils import logging
logger = logging.get_logger(__name__)


class MAGQAStreamDataset(StreamMixIn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        annos, self.annos = self.annos, []
        for anno in tqdm.tqdm(annos):
            video_uid = anno['video_uid']
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            if not anno['conversation']:
                continue
            role = anno['conversation'][0]['role']
            time = anno['conversation'][0]['time']
            video_start_time = anno.get('video_start_time', 100000000)      # video starting from here should be used as input
            content = anno['conversation'][0]['content']
            if not (role == 'user' and time > 0 and time <= duration and content):
                continue

            # 1. add random frames before the user
            fps_time = ceil_time_by_fps(time, self.frame_fps, 0, duration)
            waiting_frames = random.randint(int((fps_time - video_start_time) * self.frame_fps), int(fps_time * self.frame_fps))
            waiting_frames = max(0, min(20, waiting_frames))
            conversation = []
            if waiting_frames:
                conversation.append({'role': 'stream', 'num_frames': waiting_frames, 'learn': waiting_frames - 1})
            conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
            start_fps_time = fps_time - waiting_frames / self.frame_fps

            # 2. for loop to add message
            for message in anno['conversation'][1:]:
                role, content, time, learn, timespan = message['role'], message['content'], message['time'], message.get('learn', True), message.get('timespan', None)
                if time > duration:
                    break

                if role == 'user':
                    fps_time = ceil_time_by_fps(time, self.frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * self.frame_fps), 'learn': True})
                    conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
                else:
                    fps_time = ceil_time_by_fps(time, self.frame_fps, conversation[-1]['fps_time'], duration)
                    if fps_time > duration:
                        break
                    if fps_time > conversation[-1]['fps_time']:
                        num_frames = int((fps_time - conversation[-1]['fps_time']) * self.frame_fps)
                        conversation.append({'role': 'stream', 'num_frames': num_frames, 'learn': True})
                        # here we set informative_label = 1 for the frames that are after the middle point of the frame, but before the assistant turn point
                        # as once a response is already generated, we should not generate another one at once.
                        response_start_time = ceil_time_by_fps(np.mean([timespan[0], timespan[1]]), self.frame_fps, min_time=0, max_time=duration)
                        response_frame_num = int((time - response_start_time) * self.frame_fps) + 1
                        response_frame_num = min(response_frame_num, num_frames)
                        conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': learn, 'response_frame_num': response_frame_num})
            if not conversation:
                continue
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {video_uid: range(int(start_fps_time*self.frame_fps), int(conversation[-1]['fps_time']*self.frame_fps))}
            })

        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')

    # DEPRECATED
    def preprocess_conversation(self, conversation):
        if self.augmentation and self.is_training and len(conversation) >= 4: # 2 round
            i = random.randint(0, len(conversation) - 1) # stream, assistant, stream, ...
            if i > len(conversation) - 3:
                return [random.choice(self.user_instructions)] + conversation
            if conversation[i]['role'] == 'stream':
                i += 1 # assistant
            assert conversation[i]['role'] == 'assistant'
            correct_assistant = conversation[i]
            wrong_texts = set([turn['content'] for turn in conversation if 'assistant' == turn['role']]) - set(correct_assistant['content'])
            wrong_texts = list(wrong_texts) + ['']
            wrong_assistant = {'role': 'assistant', 'content': random.choice(wrong_texts)}
            augmented = [wrong_assistant]
            num_next_frames = conversation[i+1]['intervals'].numel()
            if num_next_frames > 1:
                if rand_bool(): # promptly fix behavior
                    frame_placeholder_with_interval = self.v_placeholders_per_frame + self.frame_interval
                    next_stream_placeholder = frame_placeholder_with_interval * (num_next_frames - 1)
                    next_intervals = torch.arange(len(frame_placeholder_with_interval), len(next_stream_placeholder)+1, len(frame_placeholder_with_interval)) - len(self.frame_interval)
                    if self.frame_interval: # last frame does not have frame interval
                        next_stream_placeholder = next_stream_placeholder[:-len(self.frame_interval)]
                    augmented += [
                        {'role': 'stream', 'content': self.v_placeholders_per_frame, 'intervals': torch.tensor([len(self.v_placeholders_per_frame)])},
                        correct_assistant,
                        {'role': 'stream', 'content': next_stream_placeholder, 'intervals': next_intervals}
                    ]
                else: # condition on video behavior
                    augmented += [
                        {'role': 'stream', 'content': conversation[i+1]['content']}
                    ]
            else:
                augmented += [conversation[i+1]]
            conversation = conversation[:i] + augmented + conversation[i+2:]
        return [random.choice(self.user_instructions)] + conversation

    def get_relevance_labels(self, conversation):
        # this label is for grounding task, no need to learn here
        return None

    def __getitem__(self, index):
        try:
            anno = self.annos[index]
            res = *super().__getitem__(
                conversation=anno['conversation'],
                load_ranges=anno['load_ranges'],
            ), index
        except Exception as e:
            logger.warning(f'Error in dataset {self.anno_file} when getting index {index}: {e}')
            logger.warning(f'Using a random data instead.')
            res = self.__getitem__(random.choice(list(range(len(self)))))
        return res


if __name__ == '__main__':
    from models.configuration_live import LiveConfigMixin
    from models.tokenization_live import build_live_tokenizer_and_update_config
    llava_config = LiveConfigMixin(frame_token_cls=True, frame_token_pooled=[3,3], frame_num_tokens=10)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)

    dataset = MAGQAStreamDataset(
        video_root='datasets/shot2story/videos_2fps_max384',
        anno_file='datasets/shot2story/annotations/livechat_train-multiturn-gpt4o-0.25_0.5-earlier.json',
        metadata_path='datasets/shot2story/videos_2fps_max384_metadata.json',
        system_prompt='This is a system prompt.',
        tokenizer=llava_tokenizer
    )
    print(len(dataset))
    print(reformat_example_for_debug(dataset[0]))
    print(reformat_example_for_debug(dataset[1]))
