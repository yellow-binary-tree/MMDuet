# reformat model output to opanai batch input, and reformat openai batch output to our sentence similarity matrix formatted eval results
import argparse, json, re, os
from tqdm import tqdm
import numpy as np

def convert_to_online_format(example):
    model_response_list = list()
    # vtimellm format
    pattern = r"From (\d+) to (\d+), (.*)"
    matches = re.findall(pattern, example['model_response'][0])
    video_length = example['video_duration']
    for match in matches:
        reply_time = (int(match[0]) / 100 * video_length + int(match[1]) / 100 * video_length) / 2
        caption = match[2]
        model_response_list.append({'time': reply_time, 'content': caption, 'role': 'assistant'})

    # timechat format
    pattern = r"(\d+\.\d+) - (\d+\.\d+)\s*seconds,\s*(.*)"
    matches = re.findall(pattern, example['model_response'][0])
    for match in matches:
        start_time, end_time, caption = float(match[0]), float(match[1]), match[2]
        model_response_list.append({'time': (start_time + end_time) / 2, 'content': caption, 'role': 'assistant'})

    if len(model_response_list) == 0:
        # the answer is not generated as grounded format; we use the entire response as 1 turn of answer, and set time = -1
        model_response_list.append({'time': -1, 'content': example['model_response'][0], 'role': 'assistant'})
    example['model_response_list'] = model_response_list


def model_output_to_openai_batch_input(
        pred_file, gold_file, output_file, is_online_model=True,
        num_examples=None, last_question_id=None
    ):
    pred_examples = [json.loads(line) for line in open(pred_file)]
    gold_examples = json.load(open(gold_file))
    gold_dict = {example['question_id']: example for example in gold_examples}
    print(f"{len(pred_examples)} pred examples to evaluate")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    f_out = open(output_file, 'w')

    for example_i, example in enumerate(tqdm(pred_examples)):
        if example_i == num_examples: break
        if is_online_model:
            pass
        else:       # convert the timechat/vtimellm generated text to online model format
            convert_to_online_format(example)

        if 'model_response_list' in example:
            if 'debug_data' in example: del example['debug_data']
            answers = [e for e in example['model_response_list'] if e['role'] == 'assistant']
            if not len(answers):
                continue
            pred_list = [e['content'] for e in answers]
            pred_time_list = [e['time'] for e in answers]

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

        question = gold_dict[example['question_id']]['conversation'][0]['content']
        for gold_answer, gold_turn_ids in gold_text_to_turn_i.items():
            for pred_answer, pred_turn_ids in pred_text_to_turn_i.items():

                # we only need to evaluate the pred answer that is in the gold span to for the in-span metric
                gold_timespan = [gold_timespan_list[i] for i in gold_turn_ids]
                pred_time = [pred_time_list[i] for i in pred_turn_ids]
                pred_time_in_gold_timespan_list = [span[0] <= time <= span[1] or time == -1 for span in gold_timespan for time in pred_time]
                if not any(pred_time_in_gold_timespan_list):
                    continue

                conversation = [
                    {"role": "system", "content": (
                        "You are an evaluator for a video question answering system. Your task is to rate the "
                        "correctness of the predicted answers against the ground truth answers. Use the following scale to assign a score:\n"
                        "- 5: Perfect match; the predicted answer is completely correct and contains all the relevant information.\n"
                        "- 4: Mostly correct; the predicted answer is largely accurate but may have minor omissions or slight inaccuracies.\n"
                        "- 3: Partially correct; the predicted answer has some correct information, but also contains significant inaccuracies or missing key points.\n"
                        "- 2: Slightly correct; the predicted answer has only a few correct elements, but most of the information is incorrect or irrelevant, or the predicted answer conflicts with the ground truth answer.\n"
                        "- 1: Incorrect; the predicted answer is entirely wrong or does not address the question at all.\n"
                        "Only reply with a number from 1 to 5, and nothing else.")
                    },
                    {"role": "user", "content": f"Question: {question}\nGround Truth Answer: {gold_answer}\nPredicted Answer: {pred_answer}"},
                ]
                custom_id = f"{example['question_id']}*{','.join(map(str, gold_turn_ids))}*{','.join(map(str, pred_turn_ids))}"
                output_example = {
                    "custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": "gpt-4o-2024-08-06", "messages": conversation}
                }
                f_out.write(json.dumps(output_example) + '\n')

        if example['question_id'] == last_question_id:
            break


def openai_batch_output_to_eval_results(
        pred_file, openai_file, gold_file, output_file, is_online_model=True,
        num_examples=None, last_question_id=None
    ):
    assert not os.path.exists(output_file), "check your filename, why do you want to create this file again?"

    openai_scores_dict = dict()
    for line in open(openai_file):
        openai_example = json.loads(line)
        question_id, gold_turn_ids, pred_turn_ids = openai_example['custom_id'].split('*')
        gold_turn_ids = list(map(int, gold_turn_ids.split(',')))
        pred_turn_ids = list(map(int, pred_turn_ids.split(',')))
        if question_id not in openai_scores_dict:
            openai_scores_dict[question_id] = dict()
        for gold_turn_id in gold_turn_ids:
            for pred_turn_id in pred_turn_ids:
                openai_scores_dict[question_id][(gold_turn_id, pred_turn_id)] = int(openai_example['response']['body']['choices'][0]['message']['content'])

    pred_examples = [json.loads(line) for line in open(pred_file)]
    gold_examples = json.load(open(gold_file))
    gold_dict = {example['question_id']: example for example in gold_examples}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    f_out = open(output_file, 'w')
    for example_i, example in enumerate(pred_examples):
        if example_i == num_examples: break
        if not is_online_model:
            convert_to_online_format(example)

        if 'model_response_list' in example:
            if 'debug_data' in example: del example['debug_data']
            answers = [e for e in example['model_response_list'] if e['role'] == 'assistant']
            if not len(answers):
                continue
            pred_list = [e['content'] for e in answers]
        gold_list = [e['content'] for e in gold_dict[example['question_id']]['answer']]

        score_matrix = np.ones((len(gold_list), len(pred_list)))
        if example['question_id'] in openai_scores_dict:
            for (gold_turn_id, pred_turn_id), score in openai_scores_dict[example['question_id']].items():
                score_matrix[gold_turn_id, pred_turn_id] = score
        example['evaluator_output'] = score_matrix.tolist()
        example['answer'] = gold_list
        example['answer_time'] = [turn['time'] for turn in gold_dict[example['question_id']]['answer']]
        f_out.write(json.dumps(example) + '\n')
        if example['question_id'] == last_question_id:
            break


def openai_send_batch(batch_input_fname, description="debug"):
    from openai import OpenAI
    client = OpenAI()
    batch_input_file = client.files.create(file=open(batch_input_fname, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    batch_metadata = client.batches.create(
        input_file_id=batch_input_file_id, 
        endpoint="/v1/chat/completions", completion_window="24h", 
        metadata={"description": description})
    print(batch_input_fname)
    print(batch_metadata)


def openai_get_batch(output_file_id, output_fname):
    from openai import OpenAI
    client = OpenAI()
    if output_file_id is not None:
        file_response = client.files.content(output_file_id)
        print(f'saving result file {output_file_id} to {output_fname}')
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)
        with open(output_fname, 'w') as f_out:
            f_out.write(file_response.text)
    else:
        print('output_file_id is None, batch not completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='batch_input')
    parser.add_argument('--file_id', type=str)
    parser.add_argument('--description', type=str)
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--openai_file', type=str)
    parser.add_argument('--gold_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--is_online_model', type=int, default=1)
    # parser.add_argument('--num_examples', type=int, default=2000)
    # parser.add_argument('--last_question_id', type=str, default='d05eUOI81LA.35.mp4')
    args = parser.parse_args()

    if args.func == 'batch_input':
        model_output_to_openai_batch_input(
            pred_file=args.pred_file, gold_file=args.gold_file, output_file=args.output_file,
            is_online_model=bool(args.is_online_model) #, num_examples=args.num_examples, last_question_id=args.last_question_id,
        )

    elif args.func == 'batch_output':
        openai_batch_output_to_eval_results(
            pred_file=args.pred_file, openai_file=args.openai_file, gold_file=args.gold_file, output_file=args.output_file,
            is_online_model=bool(args.is_online_model) #, num_examples=args.num_examples, last_question_id=args.last_question_id,
        )

    elif args.func == 'send_batch':
        openai_send_batch(batch_input_fname=args.pred_file, description=args.description)

    elif args.func == 'get_batch':
        openai_get_batch(output_file_id=args.file_id, output_fname=args.output_file)

    elif args.func == 'check_batch':
        from openai import OpenAI
        client = OpenAI()
        for task in client.batches.list(limit=6).data:
            print(task, end='\n\n')

