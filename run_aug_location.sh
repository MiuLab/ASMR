aug_image_dir="./aug_images_location"
[ ! -d "$aug_image_dir" ] && mkdir -p "$aug_image_dir"


CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
  --task_name aug \
  --aug_mode location \
  --location_file ./location.json \
  --aug_num 10 \
  --aug_times 10 \
  --llm_model_name gpt-4o-mini \
  --vision_model_name stabilityai/stable-diffusion-xl-base-1.0 \
  --output_aug_data ./aug_data/new_dataset_location.json \
  --output_aug_images ./aug_images_location \
  --instruction_file ./aug_data/instruction_location.json \


