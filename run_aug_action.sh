aug_image_dir="./aug_images_action"
[ ! -d "$aug_image_dir" ] && mkdir -p "$aug_image_dir"


CUDA_VISIBLE_DEVICES=2,3 python3 main.py \
  --task_name aug \
  --aug_mode action \
  --action_file ./new_answer.pkl \
  --aug_num 10 \
  --aug_times 10 \
  --llm_model_name gpt-4o-mini \
  --vision_model_name Salesforce/blipdiffusion \
  --output_aug_data ./aug_data/new_dataset_action.json \
  --output_aug_images ./aug_images_action \
  --instruction_file ./aug_data/instruction_action.json \


