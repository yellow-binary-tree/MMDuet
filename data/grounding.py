import json, os, shutil
import cv2
from tqdm import tqdm, trange
import math
import random

from transformers.utils import logging
from .stream import StreamMixIn
from .utils import reformat_example_for_debug, DictWithTo
logger = logging.get_logger(__name__)


class GroundingStreamDataset(StreamMixIn):
    query_templates = [
        "%s",
        "%s",
        "What segment of the video addresses the topic '%s'?",
        "At what timestamp can I find information about '%s' in the video?",
        "Can you highlight the section of the video that pertains to '%s'?",
        "Which moments in the video discuss '%s' in detail?",
        "Identify the parts that mention '%s'.",
        "Where in the video is '%s' demonstrated or explained?",
        "What parts are relevant to the concept of '%s'?",
        "Which clips in the video relate to the query '%s'?",
        "Can you point out the video segments that cover '%s'?",
        "What are the key timestamps in the video for the topic '%s'?"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        annos, self.annos = self.annos, list()
        for anno in tqdm(annos):
            video_uid = anno['video_uid']
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            conversation, current_frame = list(), 0
            conversation.append({'role': 'user', 'content': random.choice(self.query_templates) % anno['query'], 'learn': False})
            related_info = list()
            for start_time, end_time in anno['timestamps']:
                start_frame = math.floor(start_time * self.frame_fps)
                if start_frame > current_frame:
                    related_info.append({'related': False, 'num_frames': start_frame - current_frame})
                end_frame = math.floor(end_time * self.frame_fps)
                related_info.append({'related': True, 'num_frames': end_frame - start_frame})
                current_frame = end_frame
            last_frame = math.floor(duration * self.frame_fps)
            if last_frame > current_frame:
                related_info.append({'related': False, 'num_frames': last_frame - current_frame})
            conversation.append({'role': 'stream', 'num_frames': last_frame, 'learn': True, 'related': related_info})
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {video_uid: range(0, last_frame)}
            })
        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')

    def get_informative_labels(self, conversation):
        # this label is for captioning and qa task, no need to learn here
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
    llava_config = LiveConfigMixin(frame_token_cls=False, frame_token_pooled=[1,1], frame_num_tokens=1)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)

    dataset = GroundingStreamDataset(
        video_root='datasets/queryd/videos',
        anno_file='datasets/queryd/annotations/train.json',
        metadata_path='datasets/queryd/videos_metadata.json',
        system_prompt='This is a system prompt.', tokenizer=llava_tokenizer,
        frame_fps=0.5, max_num_frames=120
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        print(reformat_example_for_debug(example))
