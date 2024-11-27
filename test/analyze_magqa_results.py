
# analyze shotstory livechat results evaluated by llama or gpt4o
import json, argparse
from tqdm import tqdm
import numpy as np


def text_score_to_int(text):
    if not isinstance(text, str): return text
    return int(text[0]) if text[0] in '12345' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str)
    parser.add_argument("--model_type", type=str, default='online')
    parser.add_argument("--num_examples", type=int, default=2000)
    parser.add_argument("--baseline_all_match", type=int, default=1, 
        help="if set to 1, when a baseline model does not provide time in the pred answer, match this pred answer with all turns of gold answers. if set to 0, skip this example.")
    parser.add_argument("--pad_with_one", type=int, default=1,
        help="if the number of examples stated is less than --num_examples, add 1 as in-span score for remaining examples")
    args = parser.parse_args()
    print(args)

    num_turns_list_dedup, num_turns_list = list(), list()
    max_acc_score_list = list()
    nearest_acc_score_list = list()
    in_span_acc_score_list = list()

    for line in tqdm(open(args.fname).readlines()[:args.num_examples]):
        eval_example = json.loads(line)

        if not args.baseline_all_match:
            if eval_example['model_response_list'][0]['time'] == -1: continue

        # stat length
        sentences = [turn['content'] for turn in eval_example['model_response_list'] if turn['role'] == 'assistant']
        num_turns_list.append(len(sentences))
        num_turns_list_dedup.append(len(set(sentences)))

        # stat scores
        max_acc_score_list.append(np.mean([max([text_score_to_int(score)] for score in turn_scores) for turn_scores in eval_example['evaluator_output']]))

        example_acc_score_list = list()
        turn_time_list = [turn['time'] for turn in eval_example['model_response_list'] if turn['role'] == 'assistant']
        for score_list, answer_time in zip(eval_example['evaluator_output'], eval_example['answer_time']):
            if args.baseline_all_match:
                answer_in_span_idx = [turn_idx for turn_idx, turn_time in enumerate(turn_time_list) if (answer_time[0] <= turn_time <= answer_time[1] or turn_time == -1)]
            else:
                answer_in_span_idx = [turn_idx for turn_idx, turn_time in enumerate(turn_time_list) if answer_time[0] <= turn_time <= answer_time[1]]
            if not answer_in_span_idx:
                example_acc_score_list.append(1)
                # pass
            else:
                example_acc_score_list.append(np.mean([text_score_to_int(score_list[idx]) for idx in answer_in_span_idx]))
        if not example_acc_score_list:
            example_acc_score_list.append(1)
            # pass
        else:
            in_span_acc_score_list.append(np.mean(example_acc_score_list))

    if len(num_turns_list) < args.num_examples and args.pad_with_one:
        num_turns_list = num_turns_list + [0] * (args.num_examples - len(num_turns_list))
        num_turns_list_dedup = num_turns_list_dedup + [0] * (args.num_examples - len(num_turns_list_dedup))
        max_acc_score_list = max_acc_score_list + [1] * (args.num_examples - len(max_acc_score_list))
        in_span_acc_score_list = in_span_acc_score_list + [1] * (args.num_examples - len(in_span_acc_score_list))

    print(args.fname, len(num_turns_list))
    # latex table format output: score & turns / turns(dedup.) & \\
    print(round(np.mean(in_span_acc_score_list), 2), end=' & ')
    print(round(np.mean(num_turns_list), 2), end='/')
    print(round(np.mean(num_turns_list_dedup), 2), end=' & ')
    print('\\')
