import os, math
import cv2
import numpy as np
import torch
from test.inference import LiveInferForBenchmark


def load_video(video_file, output_fps):
    pad_color = (0, 0, 0)
    output_resolution = 384
    max_num_frames = 400

    cap = cv2.VideoCapture(video_file)
    # Get original video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = frame_count / input_fps
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = output_height = output_resolution

    output_fps = output_fps if output_fps > 0 else max_num_frames / video_duration
    num_frames_total = math.floor(video_duration * output_fps)
    frame_sec = [i / output_fps for i in range(num_frames_total)]
    frame_list, original_frame_list, cur_time, frame_index = [], [], 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            original_frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if input_width > input_height:
                # Landscape video: scale width to the resolution, adjust height
                new_width = output_resolution
                new_height = int((input_height / input_width) * output_resolution)
            else:
                # Portrait video: scale height to the resolution, adjust width
                new_height = output_resolution
                new_width = int((input_width / input_height) * output_resolution)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            # pad the frame
            canvas = cv2.copyMakeBorder(
                resized_frame,
                top=(output_height - new_height) // 2,
                bottom=(output_height - new_height + 1) // 2,
                left=(output_width - new_width) // 2,
                right=(output_width - new_width + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )
            frame_list.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
            frame_index += 1
        if len(frame_list) >= max_num_frames:
            break
        cur_time += 1 / input_fps
    cap.release()
    return torch.tensor(np.stack(frame_list)), original_frame_list


class LiveInferForDemo(LiveInferForBenchmark):
    def encode_given_query(self, query):
        self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=self.last_role == 'stream', add_stream_prompt=True, return_tensors='pt').to('cuda')
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, use_cache=True, return_dict=True)
        self.past_key_values = outputs.past_key_values
        self.last_ids = outputs.logits[:, -1:].argmax(dim=-1)
        self.last_role = 'user'

    def input_one_frame(self):
        """
        in the interactive demo, we need to input 1 frame each time this function is called.
        to ensure that user can stop the video and input user messages.
        """
        # 1. the check query step is skipped, as all user input is from the demo page

        # 2. input a frame, and update the scores list
        video_scores = self._encode_frame()
        ret = dict(frame_idx=self.frame_idx, time=round(self.video_time, 1), **video_scores)  # the frame_idx here is after self.frame_idx += 1

        # 3. check the scores, if need to generate a response
        need_response = False
        stream_end_score = sum([v for k, v in video_scores.items() if k in self.score_heads])
        self.stream_end_prob_list.append(stream_end_score)
        self.stream_end_score_sum += stream_end_score
        if isinstance(self.running_list_length, int) and self.running_list_length > 0:
            self.stream_end_prob_list = self.stream_end_prob_list[-self.running_list_length:]
        if self.stream_end_score_sum_threshold is not None and self.stream_end_score_sum > self.stream_end_score_sum_threshold:
            need_response = True
            self.stream_end_score_sum = 0
        if self.stream_end_prob_threshold is not None and stream_end_score > self.stream_end_prob_threshold:
            need_response = True

        # 4. record the responses
        if need_response:
            response = self._generate_response()
            self.num_frames_no_reply = 0
            self.consecutive_n_frames = 0
        else:
            response = None
        ret['response'] = response

        # 5. update the video time
        self.video_time += 1 / self.frame_fps

        return ret
