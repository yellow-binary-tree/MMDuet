import os, sys, re, requests, random
import json
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd

from .qvh.eval import eval_submission, load_jsonl
from .dvc.eval_dvc import eval_with_files       # for youcook2 evaluation

class CorrectnessEvaluator:
    @torch.no_grad()
    def __init__(self, llm_pretrained):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_pretrained)
        self.model = AutoModelForCausalLM.from_pretrained(llm_pretrained, torch_dtype=torch.bfloat16, device_map='auto')
        conversation = [
            {"role": "system", "content": (
                "You are an evaluator for a video question answering system. Your task is to rate the "
                "correctness of the predicted answers against the ground truth answers. Use the following scale to assign a score:\n"
                "- 5: Perfect match; the predicted answer is completely correct and contains all the relevant information.\n"
                "- 4: Mostly correct; the predicted answer is largely accurate but may have minor omissions or slight inaccuracies.\n"
                "- 3: Partially correct; the predicted answer has some correct information, but also contains significant inaccuracies or missing key points.\n"
                "- 2: Slightly correct; the predicted answer has only a few correct elements, but most of the information is incorrect or irrelevant, or the predicted answer conflicts with the ground truth answer.\n"
                "- 1: Incorrect; the predicted answer is entirely wrong or does not address the question at all.\n\n"
                "Here are some examples to guide you:")
            },
            {"role": "user", "content": "Question: What is shown about the black car?\nGround Truth Answer: At night a black car is parked in the open space with its headlights on. The lights are very dazzling.\nPredicted Answer: The car's headlights are on and dazzling."},
            {"role": "assistant", "content": "4"},

            {"role": "user", "content": "Question: What is shown in the video?\nGround Truth Answer: In the video, a group of colorful paper birds on the wall move out from the upper right corner of the camera, and then a piece of blue folded paper appears in the camera.\nPredicted Answer: The colorful paper birds are created by folding paper."},
            {"role": "assistant", "content": "2"},

            {"role": "user", "content": "Question: What is the man doing?\nGround Truth Answer: The video shows a person wearing a helmet flipping several times in the air.\nPredicted Answer: The person wearing a helmet in the background is sitting in a crouch facing the other person."},
            {"role": "assistant", "content": "3"},

            {"role": "user", "content": "Question: What is the current scene about?\nGround Truth Answer: This is a close-up of a Mercedes-Benz car on display in the showroom.\nPredicted Answer: A Mercedes-Benz car is being displayed in the showroom."},
            {"role": "assistant", "content": "5"},

            {"role": "user", "content": "Question: What was the unexpected sight in the room with the formally dressed snakes?\nGround Truth Answer: The sight of the snake on the stage talking into a microphone, with many others holding cameras with their tongues out.\nPredicted Answer: A large snake lying on its back in a room with wooden walls and furniture, surrounded by other snakes."},
            {"role": "assistant", "content": "3"},

            {"role": "user", "content": "Question: What had changed between the beginning and the end of the scene with the man in a black suit and a tie?\nGround Truth Answer: The scene changed from the man talking in the chair to the man sitting on the sofa with a woman and a pizza box, and then to the man fixing his tie and turning to look at the woman.\nPredicted Answer: The man in a black suit and tie is eating pizza."},
            {"role": "assistant", "content": "1"}
        ]

        prompt_input = self.tokenizer.apply_chat_template(conversation, return_tensors='pt', return_dict=True).to(self.model.device)
        outputs = self.model(**prompt_input, use_cache=True)
        self.prompt_past_key_values = outputs.past_key_values
        self.prompt_input_ids = prompt_input.input_ids

    @torch.no_grad()
    def evaluate(self, question, gold_answer, pred_answer):
        conversation = [
            {"role": "user", "content": f"Question: {question}\nGround Truth Answer: {gold_answer}\nPredicted Answer: {pred_answer}"},
            {"role": "assistant", "content": ""}
        ]
        new_input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors='pt').to(self.model.device)
        first_eot_index = (new_input_ids == 128009).nonzero()[0, -1]     # remove the system prompt before the user turn (i.e., the first turn) of llama tokenizer
        new_input_ids = new_input_ids[:, first_eot_index + 1:-1]      # -1 (the last token): '<|eot|>'

        all_input_ids = torch.cat([self.prompt_input_ids, new_input_ids], dim=1)
        generated_ids = self.model.generate(input_ids=all_input_ids, past_key_values=self.prompt_past_key_values, use_cache=True, max_new_tokens=32)
        generated_ids = generated_ids[:, all_input_ids.size(1):]
        decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        score = int(decoded_text[0]) if decoded_text[0] in '12345' else 1
        return score


class LlamaServerEvaluator:
    def __init__(self, url):
        self.url = url

    def evaluate(self, question, gold_answer, pred_answer):
        print('sending to llava server:', question, gold_answer, pred_answer)
        data = {"question": question, 'gold_answer': gold_answer, 'pred_answer': pred_answer}
        response = requests.post(self.url, json=data)
        decoded_text = response.json()['text']
        score = int(decoded_text[-1]) if decoded_text[-1] in '12345' else 1
        return score


def find_continuous_positive_segments(relevance_scores, min_relevance_frames):
    segments = []
    start_index = None

    for i in range(len(relevance_scores)):
        if relevance_scores[i] > 0:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None and i - start_index >= min_relevance_frames:
                segments.append((start_index, i - 1, np.mean(relevance_scores[start_index:i])))
            start_index = None

    if start_index is not None and len(relevance_scores) - start_index >= min_relevance_frames:
        segments.append((start_index, len(relevance_scores) - 1, np.mean(relevance_scores[start_index:])))
    return segments


def is_time_in_span(time, spans):
    for span in spans:
        if time >= span[0] and time <= span[1]:
            return True
    return False


def keep_longest_true_span(boolean_list):
    max_length = 0
    current_length = 0
    start_index = 0
    best_start_index = -1
    for i, value in enumerate(boolean_list):
        if value:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                best_start_index = start_index
        else:
            current_length = 0
            start_index = i + 1
    result = [False] * len(boolean_list)
    if best_start_index != -1:
        result[best_start_index:best_start_index + max_length] = [True] * max_length
    return result, max_length


def calculate_iou(pred_scores, gold_scores, threshold, pred_get_largest_span=False, debug_data=None):
    assert len(pred_scores) == len(gold_scores)
    pred_scores = [p >= threshold for p in pred_scores]
    if pred_get_largest_span:
        pred_scores, max_length = keep_longest_true_span(pred_scores)
    intersection = sum([p and g for p, g in zip(pred_scores, gold_scores)])
    union = sum([p or g for p, g in zip(pred_scores, gold_scores)])
    iou = 0 if union == 0 else intersection / union
    return iou


def calculate_iou_span(pred_span, gold_span):
    pred_start, pred_end = pred_span
    gold_start, gold_end = gold_span
    intersection = max(0, min(pred_end, gold_end) - max(pred_start, gold_start) + 1)
    union = max(pred_end, gold_end) - min(pred_start, gold_start) + 1
    return 0 if union == 0 else intersection / union


def qvh_to_charades_format(example):
    timestamps, start_clip_id = [], None
    for score, clip_id in zip(example['answer']['saliency_scores'], example['answer']['relevant_clip_ids']):
        score = max(score)
        if score < 4:
            if start_clip_id is not None:
                timestamps.append([clip_id*2, clip_id*2])
                start_clip_id = None
        else:
            if start_clip_id is None:
                start_clip_id = clip_id

    if start_clip_id is not None:
        timestamps.append([start_clip_id*2, clip_id*2+2])
    example['timestamps'] = timestamps
    return example


def smooth_pred_list(pred_list, window_size=4):
    return [np.mean(pred_list[max(0, i-window_size):min(len(pred_list), i+window_size+1)]) for i in range(len(pred_list))]


def normalize_pred_list(pred_list):
    max_num, min_num = max(pred_list), min(pred_list)
    pred_list = [(p - min_num) / (max_num - min_num) for p in pred_list]
    return pred_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='magqa')
    parser.add_argument('--llm_pretrained', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--is_online_model', type=int, default=1)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--concat_pred_list', type=bool, default=False)
    parser.add_argument('--prev_output_file', type=str, default=None)
    # correctness
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=100000000)
    # qvh_highlight
    parser.add_argument('--relevance_threshold', type=float, default=0.1)
    parser.add_argument('--min_relevance_frames', type=int, default=5)
    args = parser.parse_args()
    print(args)

    if args.func == 'magqa':
        pred_examples = [json.loads(line) for line in open(args.pred_file)]
        if args.prev_output_file is not None:
            prev_output_examples = [json.loads(line) for line in open(args.prev_output_file)]
        else:
            prev_output_examples = []

        gold_examples = json.load(open(args.gold_file))
        gold_dict = {example['question_id']: example for example in gold_examples}
        print(f"{len(pred_examples)} pred examples to evaluate")

        if args.llm_pretrained.startswith('http'):      # this is a llama model server
            evaluator = LlamaServerEvaluator(args.llm_pretrained)
        else:
            evaluator = CorrectnessEvaluator(args.llm_pretrained)

        f_out = open(args.output_file, 'w')
        acc_list = list()
        for example_i, example in enumerate(tqdm(pred_examples)):
            if example_i < args.start_idx or example_i >= args.end_idx:
                continue

            # check if this is already processed. you can change the if condition here if you want to modify prev results
            if len(prev_output_examples) > example_i:
                f_out.write(json.dumps(prev_output_examples[example_i]) + '\n')
                continue

            if args.is_online_model:
                pass        # do not need to reformat model output
            else:       # convert the timechat/vtimellm generated text to online model format
                model_response = example['model_response'][0] if isinstance(example['model_response'], list) else example['model_response']
                model_response_list = list()
                # vtimellm format
                pattern = r"From (\d+) to (\d+), (.*)"
                matches = re.findall(pattern, model_response)
                captions = list()
                video_length = example['video_duration']
                for match in matches:
                    reply_time = (int(match[0]) / 100 * video_length + int(match[1]) / 100 * video_length) / 2
                    caption = match[2]
                    model_response_list.append({'time': reply_time, 'content': caption, 'role': 'assistant'})

                # timechat format
                pattern = r"(\d+\.\d+) - (\d+\.\d+)\s*seconds,\s*(.*)"
                matches = re.findall(pattern, model_response)
                captions, start_time = list(), 0
                for match in matches:
                    start_time, end_time, caption = float(match[0]), float(match[1]), match[2]
                    reply_time = (start_time + end_time) / 2
                    model_response_list.append({'time': reply_time, 'content': caption, 'role': 'assistant'})
                
                if len(model_response_list) == 0:
                    # the answer is not generated as grounded format; we use the entire response as 1 turn of answer, and set time = -1
                    # this example can pair with any gold span
                    model_response_list.append({'time': -1, 'content': model_response, 'role': 'assistant'})
                example['model_response_list'] = model_response_list

            if 'model_response_list' in example:
                if 'debug_data' in example: del example['debug_data']
                answers = [e for e in example['model_response_list'] if e['role'] == 'assistant']
                if not len(answers):
                    continue
                pred_list = [e['content'] for e in answers]
                pred_time_list = [e['time'] for e in answers]

            if args.concat_pred_list:
                seen_preds, new_pred_list = set(), list()
                for pred in pred_list:
                    if pred.lower().strip() not in seen_preds:
                        seen_preds.add(pred.lower().strip())
                        new_pred_list.append(pred)
                pred_list = [' '.join(new_pred_list)]

            gold_list = [e['content'] for e in gold_dict[example['question_id']]['answer']]
            gold_timespan_list = [e['time'] for e in gold_dict[example['question_id']]['answer']]

            # in case that there may be some identical turns, we only need to evaluate once for them all
            pred_text_to_turn_i = dict()
            for turn_i, text in enumerate(pred_list):
                if text not in pred_text_to_turn_i:
                    pred_text_to_turn_i[text] = list()
                pred_text_to_turn_i[text].append(turn_i)

            gold_text_to_turn_i = dict()
            for turn_i, text in enumerate(gold_list):
                if text not in gold_text_to_turn_i:
                    gold_text_to_turn_i[text] = list()
                gold_text_to_turn_i[text].append(turn_i)

            score_matrix = np.ones((len(gold_list), len(pred_list)))
            question = gold_dict[example['question_id']]['conversation'][0]['content']
            for gold_content, gold_turn_ids in gold_text_to_turn_i.items():
                for pred_content, pred_turn_ids in pred_text_to_turn_i.items():
                    # we only need to evaluate the pred answer that is in the gold span to for the in-span metric
                    gold_timespan = [gold_timespan_list[i] for i in gold_turn_ids]
                    pred_time = [pred_time_list[i] for i in pred_turn_ids]
                    # the pred answer with time -1 can pair with any other span
                    pred_time_in_gold_timespan_list = [(time == -1 or span[0] <= time <= span[1]) for span in gold_timespan for time in pred_time]
                    if not any(pred_time_in_gold_timespan_list):
                        continue

                    score = evaluator.evaluate(question, gold_content, pred_content)
                    row_indices, col_indices = np.meshgrid(gold_turn_ids, pred_turn_ids)
                    score_matrix[row_indices.flatten(), col_indices.flatten()] = score
            example['evaluator_output'] = score_matrix.tolist()
            example['answer'] = gold_list
            example['answer_time'] = [turn['time'] for turn in gold_dict[example['question_id']]['answer']]     # 0926 中午 TODO: multiturn的answer time应该修改一下
            f_out.write(json.dumps(example) + '\n')

            if example_i % 10 == 0:
                f_out.flush()
        f_out.close()

    elif args.func == 'qvh_highlight':
        pred_examples = [json.loads(line) for line in open(args.pred_file)]
        gold_examples = load_jsonl(args.gold_file)
        final_results = list()

        if args.is_online_model:
            for smooth_window_size in range(0, 15):
                print("using smooth window size", smooth_window_size)

                reformatted_pred_list = list()
                for example in pred_examples:
                    frame_interval = example['debug_data'][1]['video_time'] - example['debug_data'][0]['video_time']
                    two_sec_frames = int(2 / frame_interval)
                    video_times, pred_scores = list(), list()
                    for e in example['debug_data']:
                        video_times.append(e['video_time'])
                        if 'relevance_score' in e:
                            pred_scores.append(e['relevance_score'][1])
                        else:
                            pred_scores.append(0)
                    pred_scores = smooth_pred_list(pred_scores, smooth_window_size)
                    pred_saliency_scores = [sum(pred_scores[i: i + two_sec_frames]) for i in range(0, len(pred_scores), two_sec_frames)]
                    reformatted_pred_list.append({'qid': example['question_id'], 'pred_saliency_scores': pred_saliency_scores})

                    # DEPRECATED: this used to evaluate a list of clip scores; but our model can not generate a list of clip scores
                    # relevance_scores = [p - args.relevance_threshold for p in pred_scores]
                    # relevant_windows = find_continuous_positive_segments(relevance_scores, args.min_relevance_frames)
                    # if len(relevant_windows):
                    #     max_relevant_score, min_relevant_score = max([e[2] for e in relevant_windows]), min([e[2] for e in relevant_windows])
                    #     if max_relevant_score == min_relevant_score: min_relevant_score = 0
                    #     relevant_windows = [[e[0], e[1], int((e[2] - min_relevant_score) / (max_relevant_score - min_relevant_score) * 4.99)] for e in relevant_windows]
                    #     reformatted_pred_list.append({'qid': example['question_id'], 'pred_relevant_windows': relevant_windows})

                results = eval_submission(reformatted_pred_list, gold_examples, match_number=False)
                print(results)
                final_results.append({'smooth_window_size': smooth_window_size, 'results': results})
            if args.output_file:
                json.dump(final_results, open(args.output_file, 'w'), indent=4)

        else:
            reformatted_pred_list = list()
            for example in pred_examples:
                # this is llava baseline, extract numbers from its results
                video_length = example['video_duration']
                sec_matches = re.findall(r"\d+\.?\d*", example['model_response'][0])
                if not len(sec_matches) == 2: continue
                start_sec, end_sec = float(sec_matches[0]), float(sec_matches[1])
                if 'from' in example['model_response'][0].lower() and 'to' in example['model_response'][0].lower():     # this is a vtimellm format result
                    start_sec, end_sec = start_sec / 100 * video_length, end_sec / 100 * video_length
                pred_saliency_scores = [1 if start_sec < sec < end_sec else 0 for sec in range(0, int(video_length), 2)]
                reformatted_pred_list.append({'qid': example['question_id'], 'pred_saliency_scores': pred_saliency_scores})
            results = eval_submission(reformatted_pred_list, gold_examples, match_number=False)
            print(results)


    elif args.func == 'grounding':
        pred_examples = [json.loads(line) for line in open(args.pred_file)]
        gold_examples = json.load(open(args.gold_file))
        if 'answer' in gold_examples[0] and 'saliency_scores' in gold_examples[0]['answer']:
            # this is a qvh dataset, convert it to charades format
            gold_examples = [qvh_to_charades_format(e) for e in gold_examples]
        gold_examples = {e['question_id']: e for e in gold_examples}
        final_results = list()

        if args.is_online_model:
            for smooth_window_size in range(0, 15):
                print("using smooth window size", smooth_window_size)
                iou_scores_list_dict = {threshold: list() for threshold in np.arange(0.30, 0.71, 0.02)}
                for pred_example in pred_examples:
                    gold_example = gold_examples[pred_example['question_id']]
                    video_times, pred_scores = list(), list()
                    for e in pred_example['debug_data']:
                        video_times.append(e['video_time'])
                        if 'relevance_score' in e:
                            pred_scores.append(e['relevance_score'][1])
                        else:
                            pred_scores.append(0)

                    pred_scores = smooth_pred_list(pred_scores, smooth_window_size)
                    pred_scores = normalize_pred_list(pred_scores)
                    gold_scores = [is_time_in_span(time, gold_example['timestamps']) for time in video_times]
                    for threshold in iou_scores_list_dict:
                        iou = calculate_iou(pred_scores, gold_scores, threshold, debug_data=pred_example['question_id'])
                        iou_scores_list_dict[threshold].append(iou)

                for threshold in iou_scores_list_dict:
                    mean_iou = np.mean(iou_scores_list_dict[threshold]) * 100
                    recall_0_3 = np.mean([e >= 0.3 for e in iou_scores_list_dict[threshold]]) * 100
                    recall_0_5 = np.mean([e >= 0.5 for e in iou_scores_list_dict[threshold]]) * 100
                    recall_0_7 = np.mean([e >= 0.7 for e in iou_scores_list_dict[threshold]]) * 100
                    print(f'score threshold = {threshold:.2f}: {mean_iou:.2f}/{recall_0_3:.2f}/{recall_0_5:.2f}/{recall_0_7:.2f}')
                    final_results.append({'smooth_window_size': smooth_window_size, 'threshold': threshold, 'scores': [mean_iou, recall_0_3, recall_0_5, recall_0_7]})

                best_among_all_thres = [max([iou_list[i] for iou_list in iou_scores_list_dict.values()]) for i in range(len(pred_examples))]
                mean_iou = np.mean(best_among_all_thres) * 100
                recall_0_3 = np.mean([e >= 0.3 for e in best_among_all_thres]) * 100
                recall_0_5 = np.mean([e >= 0.5 for e in best_among_all_thres]) * 100
                recall_0_7 = np.mean([e >= 0.7 for e in best_among_all_thres]) * 100
                print(f'best among all thresholds: {mean_iou:.2f}/{recall_0_3:.2f}/{recall_0_5:.2f}/{recall_0_7:.2f}')
            if args.output_file:
                json.dump(final_results, open(args.output_file, 'w'), indent=4)

        else:
            iou_scores = list()
            for example in pred_examples:
                # this is llava baseline, extract numbers from its results
                gold_example = gold_examples[example['question_id']]
                sec_matches = re.findall(r"\d+\.?\d*", example['model_response'][0])
                if not len(sec_matches) == 2: continue
                start_sec, end_sec = float(sec_matches[0]), float(sec_matches[1])
                if 'from' in example['model_response'][0].lower() and 'to' in example['model_response'][0].lower():
                    # this is a vtimellm formatt output
                    video_length = example['video_duration']
                    start_sec, end_sec = start_sec / 100 * video_length, end_sec / 100 * video_length
                iou_scores.append(calculate_iou_span((start_sec, end_sec), gold_example['timestamps'][0]))

            mean_iou = np.mean(iou_scores) * 100
            recall_0_3 = np.mean([e >= 0.3 for e in iou_scores]) * 100
            recall_0_5 = np.mean([e >= 0.5 for e in iou_scores]) * 100
            recall_0_7 = np.mean([e >= 0.7 for e in iou_scores]) * 100
            print(f'score: {mean_iou:.2f}/{recall_0_3:.2f}/{recall_0_5:.2f}/{recall_0_7:.2f}')


    elif args.func == 'dense_captioning':
        pred_examples = [json.loads(line) for line in open(args.pred_file)]
        gold_examples = json.load(open(args.gold_file))

        pred_out, gold_out = dict(), list()
        for pred_example in pred_examples:
            if args.is_online_model:
                captions, prev_sent, start_time, end_time = list(), None, None, None
                for turn in pred_example['model_response_list']:
                    if turn['role'] == 'user': continue
                    if turn['content'] != prev_sent:
                        if start_time is not None:
                            captions.append({'timestamp': [start_time, end_time], 'caption': prev_sent})
                        prev_sent, start_time, end_time = turn['content'], end_time, turn['time']
                    else:
                        end_time = turn['time']

                if start_time is not None:
                    captions.append({'timestamp': [start_time, end_time], 'caption': prev_sent})
                pred_out[str(pred_example['question_id'])] = captions
            else:
                model_response = pred_example['model_response'][0] if isinstance(pred_example['model_response'], list) else pred_example['model_response']
                if 'vtimellm' in args.pred_file:
                    # this is a vtimellm format response
                    pattern = r"From (\d+) to (\d+), (.*)"
                    matches = re.findall(pattern, model_response)
                    captions = list()
                    video_length = pred_example['video_duration']
                    for match in matches:
                        captions.append({'timestamp': [int(match[0]) / 100 * video_length, int(match[1]) / 100 * video_length], 'caption': match[2]})
                    pred_out[str(pred_example['question_id'])] = captions
                else:
                    # this is a timechat format response
                    pattern = r"(\d+\.\d+) - (\d+\.\d+)\s*seconds,\s*(.*)"
                    # pattern = r"(\d+\.\d+)\s*seconds,\s*(.*)"
                    matches = re.findall(pattern, model_response)
                    captions = list()
                    for match in matches:
                        start_time, end_time, action = float(match[0]), float(match[1]), match[2]
                        captions.append({'timestamp': [start_time, end_time], 'caption': action})
                        start_time = end_time
                    pred_out[str(pred_example['question_id'])] = captions

        # youcook2 dense captioning evaluation
        for gold_example in gold_examples:
            if str(gold_example['question_id']) not in pred_out: continue
            segments = [turn['time'] for turn in gold_example['answer']]
            answer_list = [turn['content'] for turn in gold_example['answer']]
            answer_list = [ans.replace('. ', ', ') for ans in answer_list]
            pure_cap = '. '.join(answer_list)
            gold_out.append({'image_id': str(gold_example['question_id']), 'segments': segments, 'pure_cap': pure_cap})

        base_folder = os.path.dirname(args.pred_file)
        os.makedirs(os.path.join(base_folder, 'tmp'), exist_ok=True)
        temp_pred_fname = os.path.join(base_folder, f'tmp/{os.path.basename(args.pred_file)}')
        temp_gold_fname = os.path.join(base_folder, 'tmp/val.json')
        with open(temp_pred_fname, 'w') as f:
            json.dump(pred_out, f)
        with open(temp_gold_fname, 'w') as f:
            json.dump({"annotations": gold_out}, f)
        eval_with_files(temp_pred_fname, temp_gold_fname)
