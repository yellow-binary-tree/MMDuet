output_dir=outputs/mmduet
mkdir -vp  ${output_dir}/eval

thres_sum=2

# --------------------
# run inference
# --------------------
python -u -m test.inference \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${output_dir} \
    --input_dir datasets/youcook2/videos --frame_fps 0.5 --max_num_frames 200 \
    --test_fname datasets/youcook2/annotations/val-random_prompt.json \
    --stream_end_score_sum_threshold ${thres_sum} --remove_assistant_turns true \
    --output_fname ${output_dir}/eval/youcook2_val-thres_sum_${thres_sum}-rm_ass_turns-pred.json \
    > ${output_dir}/eval/youcook2_val-thres_sum_${thres_sum}-rm_ass_turns-pred.log 2>&1 &
wait

# --stream_end_score_sum_threshold is the theshold of the sum of informative score, 
#     when the sum reaches this theshold, assistant generates a response
# --remove_assistant_turns is the rm. ass. turns trick that do not add previous assistant-generated responses in conversation context.

# --------------------
# evaluate the model generated results
# --------------------
python -m test.evaluate --func dense_captioning \
    --pred_file ${output_dir}/eval/youcook2_val-thres_sum_${thres_sum}-rm_ass_turns-pred.json \
    --gold_file datasets/youcook2/annotations/val-random_prompt.json \
    > ${output_dir}/eval/youcook2_val-thres_sum_${thres_sum}-rm_ass_turns-eval.log 2>&1 &
