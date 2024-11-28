output_dir=outputs/mmduet
mkdir -vp  ${output_dir}/eval


# --------------------
# run inference
# --------------------
python -u -m test.inference --grounding_mode true \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov --bf16 true \
    --lora_pretrained ${output_dir} \
    --stream_end_prob_threshold 1 \
    --input_dir datasets/qvh/videos --frame_fps 1 --max_num_frames 400 \
    --test_fname datasets/qvh/annotations/highlight_val-random_prompt.json \
    --output_fname ${output_dir}/eval/qvh_val-random_prompt-pred.json \
    > ${output_dir}/eval/qvh_val-random_prompt-pred.log 2>&1 &
wait


# --------------------
# evaluate
# --------------------
python -u -m test.evaluate --func qvh_highlight  \
    --pred_file ${output_dir}/eval/qvh_val-random_prompt-pred.json \
    --gold_file datasets/qvh/annotations/highlight_val_release.jsonl \
    --output_file ${output_dir}/eval/qvh_val-random_prompt-eval.json \
    > ${output_dir}/eval/qvh_val-random_prompt-eval.log 2>&1 &
