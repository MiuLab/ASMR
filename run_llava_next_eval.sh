 

python3 main.py \
  --task_name eval \
  --task_type video \
  --eval_data ./question_utter_desc.jsonl \
  --eval_model lmms-lab/llava-onevision-qwen2-7b-ov \
  --output_eval ./result/answer_utter_desc_llavanext_gpt3.jsonl \
  --matching_mode gpt3 \
  --output_score ./result/result_utter_desc_llavanext_gpt3.json \
  #--label_mode high

