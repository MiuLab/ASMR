

python3 main.py \
  --task_name finetune \
  --output_aug_data aug_data/new_dataset_both.json \
  --output_aug_images aug_images_both \
  --instruction_file aug_data/instruction_both.json \
  --finetune_model_name liuhaotian/llava-v1.6-vicuna-7b \
  --finetune_output_dir LLaVA/checkpoints/llava-v1.6-vicuna-7b-both-lora \
  --merged_model_dir LLaVA/checkpoints/llava-v1.6-vicuna-7b-both-merged \
  --finetune_both \


