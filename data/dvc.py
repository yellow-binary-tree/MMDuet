import tqdm, random
import numpy as np

from .stream import StreamMixIn
from .utils import ceil_time_by_fps, DictWithTo, reformat_example_for_debug
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DenseVideoCaptioningStreamDataset(StreamMixIn):
    instructions = [
        {"role": "user", "content": "Please concisely narrate the video in real time."},
        {"role": "user", "content": "Help me to illustrate my view in short."},
        {"role": "user", "content": "Please simply describe what do you see."},
        {"role": "user", "content": "Continuously answer what you observed with simple text."},
        {"role": "user", "content": "Do concise real-time narration."},
        {"role": "user", "content": "Hey assistant, do you know the current video content? Reply me concisely."},
        {"role": "user", "content": "Simply interpret the scene for me."},
        {"role": "user", "content": "What can you tell me about? Be concise."},
        {"role": "user", "content": "Use simple text to explain what is shown in front of me."},
        {"role": "user", "content": "What is the action now? Please response in short."},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        annos, self.annos = self.annos, []
        for video_uid, _annotation_uid_narrations in tqdm.tqdm(annos.items(), desc=self.anno_file):
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            for narrations in _annotation_uid_narrations.values():
                if not narrations:
                    continue
                start_time = ceil_time_by_fps(0, self.frame_fps, min_time=0, max_time=duration)
                conversation = []
                last_time = start_time
                last_text = None
                for narration in narrations:
                    if last_time >= duration:
                        break
                    text = narration['text']
                    learn = narration.get('learn', True)
                    if text == last_text:
                        continue
                    time = ceil_time_by_fps(narration['time'], self.frame_fps, min_time=0, max_time=duration)
                    if time == last_time: # since we have sorted and ceiled, so directly replace, this time is more close
                        conversation[-1]['content'] = text
                    else: # time > last_time
                        num_frames = int((time - last_time) * self.frame_fps)
                        # here we set informative_label = 1 for the frames that are after the middle point of the frame, but before the assistant turn point
                        # as once a response is already generated, we should not generate another one at once.
                        response_start_time = ceil_time_by_fps(np.mean([narration['timespan'][0], narration['timespan'][1]]), self.frame_fps, min_time=0, max_time=duration)
                        response_frame_num = int((time - response_start_time) * self.frame_fps) + 1
                        conversation.extend([
                            {"role": "stream", 'num_frames': num_frames, 'learn': True},
                            {"role": "assistant", "content": text, 'learn': learn, 'response_frame_num': response_frame_num},
                        ])
                    last_time = time
                    last_text = text
                if not conversation:
                    continue
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {video_uid: range(int(start_time*self.frame_fps), int(last_time*self.frame_fps))}
                })
        print(f'Dataset {self.__class__.__name__} has {len(self)} examples. Example data: {reformat_example_for_debug(self[0])}')

    def preprocess_conversation(self, conversation):
        return [random.choice(self.instructions)] + conversation

    def get_relevance_labels(self, conversation):
        # this label is for grounding task, no need to learn here
        return None

    def __getitem__(self, index):
        try:
            anno = self.annos[index]
            return *super().__getitem__(
                conversation=self.preprocess_conversation(anno['conversation']),
                load_ranges=anno['load_ranges'],
            ), index
        except Exception as e:
            logger.warning(f'Error in dataset {self.anno_file} when getting index {index}: {e}')
            logger.warning(f'Using a random data instead.')
            return self.__getitem__(random.choice(list(range(len(self)))))


if __name__ == '__main__':
    from models.configuration_live import LiveConfigMixin
    from models.tokenization_live import build_live_tokenizer_and_update_config
    llava_config = LiveConfigMixin(frame_token_cls=True, frame_token_pooled=[3,3], frame_num_tokens=10)
    llava_tokenizer = build_live_tokenizer_and_update_config('lmms-lab/llava-onevision-qwen2-7b-ov', llava_config)

    dataset = DenseVideoCaptioningStreamDataset(
        video_root='datasets/shot2story/videos_2fps_max384',
        anno_file='datasets/shot2story/annotations/narration_stream_train-human_anno-0.25_0.5_earlier.json',
        metadata_path='datasets/shot2story/videos_2fps_max384_metadata.json',
        system_prompt='This is a system prompt.',
        tokenizer=llava_tokenizer,
        frame_fps=2, max_num_frames=100
    )

    print('length of the dataset:', len(dataset))
    for i in range(0, min(1000, len(dataset)), 20):
        example = dataset[i]
        print(reformat_example_for_debug(example))
