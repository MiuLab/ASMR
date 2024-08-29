

python3 main.py \
  --task_name finetune \
  --output_aug_data aug_data/new_dataset_location.json \
  --output_aug_images aug_images_location \
  --instruction_file aug_data/instruction_location.json \
  --finetune_model_name liuhaotian/llava-v1.6-vicuna-7b \
  --finetune_output_dir llava-v1.6-vicuna-7b-location-lora \
  --merged_model_dir llava-v1.6-vicuna-7b-location-merged \

