 

python3 main.py \
  --task_name eval \
  --eval_data ./question_utter.jsonl \
  --eval_model gpt-4o-mini \
  --output_eval ./result/answer_utter_gpt_gpt3.jsonl \
  --matching_mode gpt3 \
  --output_score ./result/result_utter_gpt_gpt3.json \
  #--label_mode high
