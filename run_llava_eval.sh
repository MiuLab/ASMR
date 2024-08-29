 

python3 main.py \
  --task_name eval \
  --eval_data ./question_utter.jsonl \
  --eval_model liuhaotian/llava-v1.6-vicuna-13b \
  --output_eval ./result/answer_utter_llava13b_gpt3.jsonl \
  --matching_mode gpt3 \
  --output_score ./result/result_utter_llava13b_gpt3.json \
  #--label_mode high

