import torch, os, json, tqdm, math, random, cv2
import numpy as np
from transformers import PreTrainedTokenizer
import torch.distributed as dist
import multiprocessing as mp

from .utils import rand_bool, resize_and_pad_frame


def get_all_files(directory):
    relative_file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the relative path by removing the directory part from the absolute path
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            relative_file_list.append(relative_path)
    return relative_file_list


def get_video_duration_and_fps(args):
    file, video_root = args
    path = os.path.join(video_root, file)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    return file, {'duration': duration, 'fps': fps, 'path': path, 'frame_count': frame_count}


class StreamMixIn(torch.utils.data.Dataset):
    def __init__(self,
                 video_root: str = None, anno_file: str = None, metadata_path: str = None, frame_fps: float = 2, frame_size: int = 384,
                 system_prompt: str = None, augmentation: bool = False,
                 max_num_frames: int = 128, tokenizer: PreTrainedTokenizer = None, skip_video=False, **kwargs):
        super().__init__()
        self.video_root = video_root
        self.anno_file = anno_file
        self.metadata_path = metadata_path
        self.frame_fps = frame_fps
        self.frame_size = frame_size
        self.system_prompt = system_prompt if system_prompt is not None else "A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant."
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        self.skip_video = skip_video     # used in text-only scenarios
        self.metadata = self.get_metadata()
        self.annos = self.get_annos()

    def __len__(self):
        return len(self.annos)

    def get_annos(self) -> dict:
        anno_path = os.path.join(self.anno_file)
        assert os.path.exists(anno_path)
        return json.load(open(anno_path))

    def max_frames_clip(self, conversation: list[dict], load_ranges: dict[str, range], max_num_frames: int):
        cum_num_frames = 0
        for i, message in enumerate(conversation):
            if message['role'] == 'stream':
                if cum_num_frames + message['num_frames'] >= max_num_frames:
                    if cum_num_frames < max_num_frames:
                        # crop this video stream to fewer frames
                        conversation[i]['num_frames'] = max_num_frames - cum_num_frames
                        conversation = conversation[:i+1]
                    else:
                        conversation = conversation[:i]
                    load_ranges = {path: range(ranger.start, ranger.start + max_num_frames) for path, ranger in load_ranges.items()}
                    break
                cum_num_frames += message['num_frames']
        return conversation, load_ranges

    def get_metadata(self):
        if os.path.exists(self.metadata_path):
            print(f'load {self.metadata_path}...')
            metadata = json.load(open(self.metadata_path))
        else:
            metadata = {}
            if not dist.is_initialized() or dist.get_rank() == 0:
                # only the main process needs to prepare metadata
                files = get_all_files(self.video_root)
                with mp.Pool(20) as pool:
                    results = list(tqdm.tqdm(pool.imap(
                        get_video_duration_and_fps, [(file, self.video_root) for file in files]),
                        total=len(files), desc=f'prepare {self.metadata_path}...'))
                for key, value in results:
                    metadata[key] = value
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                if dist.is_initialized():
                    dist.barrier()
            else:
                dist.barrier()
                metadata = json.load(open(self.metadata_path))
        return metadata

    def load_video(self, file):
        video_metadata = self.metadata[file]
        # load the frames, and downsample to self.frame_fps
        cap = cv2.VideoCapture(video_metadata['path'])
        num_frames_total = math.floor(video_metadata['duration'] * self.frame_fps)
        frame_sec = [i / self.frame_fps for i in range(num_frames_total)]
        frames, cur_time, frame_index = [], 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
                frame = resize_and_pad_frame(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_index += 1
            cur_time += 1 / video_metadata['fps']
        cap.release()
        frames = np.array(frames)  # shape will be (T, H, W, C)
        frames = np.transpose(frames, (0, 3, 1, 2))  # Change to (T, C, H, W)
        return torch.tensor(frames)

    def get_informative_labels(self, conversation):
        informative_labels = list()
        for i, turn in enumerate(conversation):
            if turn['role'] == 'stream' and turn['num_frames'] > 0:
                if turn['learn']:
                    if i != len(conversation) - 1:
                        next_turn = conversation[i + 1]
                        response_frame_num = next_turn.get('response_frame_num', 1)
                        next_role = next_turn['role']
                    else:
                        response_frame_num = 1
                        next_role = None
                    informative_labels += [0] * (turn['num_frames'] - response_frame_num)
                    informative_labels += [int(next_role == 'assistant')] * response_frame_num
                else:
                    informative_labels += [-100] * turn['num_frames']
        return informative_labels

    def get_relevance_labels(self, conversation):
        relevance_labels = list()
        for turn in conversation:
            if turn['role'] == 'stream' and turn['num_frames'] > 0:
                if turn['learn']:
                    for related_info in turn['related']:
                        relevance_labels += [int(related_info['related'])] * related_info['num_frames']
                else:
                    relevance_labels += [-100] * turn['num_frames']
        return relevance_labels

    def __getitem__(self, *, conversation: list[dict], load_ranges: dict[str, range] | torch.Tensor = None, add_generation_prompt=False, **kwargs):
        # 1. load videos
        if self.skip_video:
            frames = torch.tensor([])
        elif isinstance(load_ranges, torch.Tensor):
            frames = load_ranges
        elif load_ranges is not None:
            conversation, load_ranges = self.max_frames_clip(conversation, load_ranges, self.max_num_frames)
            # after max_frames_clip, sometimes there may be no conversation left due to the conversations are too late.
            # we also need to keep this kind of data, as no conversation can also be a real-time situation
            ranges = [self.load_video(path)[ranger] for path, ranger in load_ranges.items()]
            frames = torch.cat(ranges)
        else:
            frames = torch.tensor([])

        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)

        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        # check if the number of frames in video and text is equal
        if not self.skip_video:
            num_frames_in_video = len(frames)
            num_frames_in_text = sum([turn['num_frames'] for turn in conversation if turn['role'] == 'stream'])
            assert num_frames_in_video == num_frames_in_text, f"num_frames_in_video: {num_frames_in_video}, num_frames_in_text: {num_frames_in_text}"

        # 4. get response labels or related labels according to subclass
        # the default logic is written in this class. if do not want to learn with this label, you can override in subclass with `return None`
        informative_labels, relevance_labels = self.get_informative_labels(conversation), self.get_relevance_labels(conversation)
        if not self.skip_video and informative_labels is not None:
            assert len(informative_labels) >= len(frames), f"len(informative_labels): {len(informative_labels)}, len(frames): {len(frames)}"
            informative_labels = informative_labels[:len(frames)]
        if not self.skip_video and relevance_labels is not None:
            assert len(relevance_labels) >= len(frames), f"len(relevance_labels): {len(relevance_labels)}, len(frames): {len(frames)}"
            relevance_labels = relevance_labels[:len(frames)]

        return text, frames, learn_ranges, informative_labels, relevance_labels
