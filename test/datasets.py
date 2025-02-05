import os, json, math, random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class FastAndAccurateStreamingVideoQADataset(Dataset):
    """
    Dataset class for Fast and Accurate Streaming Video Question Answering Benchmarks
    """
    def __init__(self, data_file, video_base_folder, start_idx=0, end_idx=None,
                 output_fps=2, output_resolution=384, max_num_frames=100, time_instruction_format=None,
                 system_prompt="A multimodal AI assistant is helping users with some activities."
                 " Below is their conversation, interleaved with the list of video frames received by the assistant."):
        """
        set output_fps = 'auto' to always load "max_num_fraems" frames from the video.a
        this is used when the lengths of videos vary significantly in the test set.
        """
        self.data = json.load(open(data_file))[start_idx: end_idx]
        self.video_base_folder = video_base_folder
        self.output_fps = output_fps
        self.output_resolution = output_resolution
        self.max_num_frames = max_num_frames
        self.pad_color = (0, 0, 0)
        self.system_prompt = system_prompt
        self.time_instruction_format = time_instruction_format        # provide frame time for traditional video llms
        print(f'loaded {len(self)} samples from {data_file}. Example data:')
        print(self[0])
        print(self[random.randint(0, len(self)-1)])

    def load_video(self, video_file):
        video_file = os.path.join(self.video_base_folder, video_file)
        cap = cv2.VideoCapture(video_file)
        # Get original video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = frame_count / input_fps
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_width = output_height = self.output_resolution

        output_fps = self.output_fps if self.output_fps > 0 else self.max_num_frames / video_duration
        num_frames_total = math.ceil(video_duration * output_fps)
        frame_sec = [i / output_fps for i in range(num_frames_total)]
        frame_list, cur_time, frame_index = [], 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
                if input_width > input_height:
                    # Landscape video: scale width to the resolution, adjust height
                    new_width = self.output_resolution
                    new_height = int((input_height / input_width) * self.output_resolution)
                else:
                    # Portrait video: scale height to the resolution, adjust width
                    new_height = self.output_resolution
                    new_width = int((input_width / input_height) * self.output_resolution)
                resized_frame = cv2.resize(frame, (new_width, new_height))
                # pad the frame
                canvas = cv2.copyMakeBorder(
                    resized_frame,
                    top=(output_height - new_height) // 2,
                    bottom=(output_height - new_height + 1) // 2,
                    left=(output_width - new_width) // 2,
                    right=(output_width - new_width + 1) // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=self.pad_color
                )
                frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
                frame_index += 1
            if len(frame_list) >= self.max_num_frames:
                break
            cur_time += 1 / input_fps
        cap.release()

        if self.time_instruction_format == 'timechat':
            frame_sec_str = ",".join([f"{i:.2f}s" for i in frame_sec])
            time_instruciton = f"The video lasts for {video_duration:.2f} seconds, and {len(frame_list)} frames are uniformly sampled from it. These frames are located at {frame_sec_str}.Please answer the following questions related to this video."
            return torch.tensor(np.stack(frame_list)), output_fps, video_duration, time_instruciton
        elif self.time_instruction_format == 'vtimellm':
            time_instruciton = f"This is a video with {len(frame_list)} frames."
            return torch.tensor(np.stack(frame_list)), output_fps, video_duration, time_instruciton
        return torch.tensor(np.stack(frame_list)), output_fps, video_duration

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        try:
            conversation = example['conversation']
            question_id = example['question_id']
            if self.time_instruction_format is None:
                video_frames, output_fps, video_duration = self.load_video(example['video'])
            else:
                video_frames, output_fps, video_duration, time_instruction = self.load_video(example['video'])
                conversation[0]['content'] = time_instruction + '\n' + conversation[0]['content']
            conversation.insert(0, {"role": "system", "content": self.system_prompt})
            return question_id, video_frames, conversation, output_fps, video_duration
        except Exception as e:
            print(f"error loading {example['question_id']} due to exception {e}, this example will be skipped")
            return None, None, None, None, None


class StreamingVideoQADatasetWithGenTime(FastAndAccurateStreamingVideoQADataset):
    def __getitem__(self, idx):
        example = self.data[idx]
        try:
            conversation = example['conversation']
            question_id = example['question_id']
            video_frames, output_fps, video_duration = self.load_video(example['video'])
            conversation.insert(0, {"role": "system", "content": self.system_prompt})
            gen_time_list = [i['time'][1] for i in example['answer']]
            return question_id, video_frames, conversation, output_fps, video_duration, gen_time_list
        except Exception as e:
            print(f"error loading {example['question_id']} due to exception {e}, this example will be skipped")
            return None, None, None, None, None
