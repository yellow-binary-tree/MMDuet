output_dir=outputs/mmduet
mkdir -vp  ${output_dir}/eval

thres=0.5

# --------------------
# run inference
# --------------------
python -u -m test.inference \
--llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
--lora_pretrained ${output_dir} \
--input_dir datasets/shot2story/videos --frame_fps 2 --max_num_frames 400 \
--test_fname datasets/shot2story/annotations/magqa_test.json \
--stream_end_prob_threshold ${thres} --score_heads "informative_score,relevance_score" \
--remove_assistant_turns true \
--output_fname ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-pred.json \
> ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-pred.log 2>&1 &
wait

# --stream_end_prob_threshold when the scores reach this theshold, stop the video stream and start a assistant turn.
# --score_heads "informative_score,relevance_score" the two scores are added when comparing with stream_end_prob_threshold
# --remove_assistant_turns is the rm. ass. turns trick that do not add previous assistant-generated responses in conversation context.


# --------------------
# use LLaMA to evaluate and get the in-span score
# --------------------
# 1. calculate similarities between pred and gold answers
python -u -m test.evaluate --func magqa \
--llm_pretrained meta-llama/Meta-Llama-3.1-70B-Instruct \
--gold_file datasets/shot2story/annotations/magqa_test.json \
--pred_file ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-pred.json \
--output_file ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-llama_score-eval.json \
> ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-llama_score-eval.log 2>&1 &
wait

# 2. analyze the LLaMA-calculated scores to get the final in-span score
python test/analyze_magqa_results.py \
    --fname ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-llama_score-eval.json


# --------------------
# use GPT-4o to evaluate and get the in-span score
# --------------------
# we use OpenAI batch api to calculate the pred-gold answer similarity to save money and time

# 0. set the openai key
export OPENAI_API_KEY="your_openai_api_key"

# 1. create batch input
python test/openai_batch.py --func batch_input \
    --pred_file ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-pred.json \
    --gold_file datasets/shot2story/annotations/magqa_test.json \
    --output_file ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_input.jsonl
wc -l ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_input.jsonl

# 2. submit this batch
python test/openai_batch.py --func send_batch \
    --pred_file ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_input.jsonl \
    --description "xxx magqa test set evaluate"         # you can change to other descriptions for this task

# 3. wait until the Batch API service finish the evaluation process. You can check the progress by running the following command and get output_file_id
python test/openai_batch.py --func check_batch

# 4. download the similarity results
python test/openai_batch.py --func get_batch \
    --file_id OUTPUT_FILE_ID_YOU_GOT_IN_THE_LAST_STEP \
    --output_file ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_output.jsonl
wc -l ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_output.jsonl

# 5. reformat the results to the format similar to LLaMA output:
python test/openai_batch.py --func batch_output \
    --pred_file ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-pred.json \
    --gold_file datasets/shot2story/annotations/magqa_test.json \
    --openai_file ${output_dir}/eval/openai/magqa_test-thres_${thres}-rm_ass_turn-pred-batch_output.jsonl \
    --output_file ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-gpt4o_score-eval.json
wc -l ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-gpt4o_score-eval.json

# 6. analyze the GPT-4o-calculated scores to get the final in-span score
python test/analyze_magqa_results.py \
    --fname ${output_dir}/eval/magqa_test-thres_${thres}-rm_ass_turn-gpt4o_score-eval.json
