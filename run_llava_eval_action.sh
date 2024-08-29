 

python3 main.py \
  --task_name eval \
  --is_finetuned \
  --eval_data question_utter_desc.jsonl \
  --merged_model_dir LLaVA/checkpoints/llava-v1.6-vicuna-7b-action-merged \
  --output_eval ./result/answer_utter_desc_llava7b_gpt3_action.jsonl \
  --matching_mode gpt3 \
  --output_score ./result/result_utter_desc_llava7b_gpt3_action.json \
  #--label_mode high
