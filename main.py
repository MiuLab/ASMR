import json
import os
import subprocess
from util import gen_instruct_format, combine_both_aug_data
from similarity import scoring
import argparse
from generate import location_augment, action_augment
from eval import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="augment robotic life-support sceraio data")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        required=True,
        choices=["aug", "finetune", "eval"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default='image',
        required=True,
        choices=["image", "video"],
        help="The name of the type of the task.",
    )
    parser.add_argument(
        "--location_file", type=str, default=None, help="A file containing all locations."
    )
    parser.add_argument(
        "--action_file", type=str, default=None, help="A file containing all actions."
    )
    parser.add_argument(
        "--aug_num",
        type=int,
        default=10,
        help="How many examples to generate for each location or action",
    )
    parser.add_argument(
        "--aug_times",
        type=int,
        default=10,
        help="How many times to generate for all locations",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default='gpt-4o-mini',
        help="model to use for utterance augmentation"
    )
    parser.add_argument(
        "--vision_model_name",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help="model to use for environment image augmentation"
    )
    parser.add_argument(
        "--aug_mode",
        type=str,
        help="Specify which method to augment the scenario",
        choices=["action", "location"]
    )
    parser.add_argument(
        "--output_aug_data",
        type=str,
        default='./aug_data/new_dataset.json',
        help="augmented utterance data path"
    )
    parser.add_argument(
        "--output_aug_images",
        type=str,
        default='./aug_images',
        help="augmented image data path"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default='./question_utter.jsonl',
        help="testing dataset"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default='gpt-4o-mini',
        help="model name for evaluation"
    )
    parser.add_argument(
        "--is_finetuned",
        action="store_true",
        help="if passed, finetune the model using audmented data."
    )
    parser.add_argument(
        "--output_eval",
        type=str,
        default='./result/answer_utter.jsonl',
        help="output of evaluation data"
    )
    parser.add_argument(
        "--matching_mode",
        type=str,
        help="Specify which type to match the label",
        default='gpt3',
        choices=["gpt3", "sbert"]
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        help="Specify which type of labels",
        default='low',
        choices=["low", "high"]
    )
    parser.add_argument(
        "--output_score",
        type=str,
        default='./result/result.json',
        help="output of scoring result"
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        default='./instruction_action.json',
        help="output of instruction data"
    )
    parser.add_argument(
        "--finetune_model_name",
        type=str,
        default='liuhaotian/llava-v1.6-vicuna-13b',
        help="llava model we use to finetune"
    )
    parser.add_argument(
        "--finetune_output_dir",
        type=str,
        default='../checkpoints/liuhaotian/llava-v1.6-vicuna-13b-lora',
        help="finetuned lora model "
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        default='../checkpoints/liuhaotian/llava-v1.6-vicuna-13b-merged',
        help="merged finetuned model"
    )
    parser.add_argument(
        "--finetune_both",
        action="store_true",
        help="if passed, finetune the model using both location and action audmented data."
    )
    


    args = parser.parse_args()


    return args 

def main():
    args = parse_args()
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    if not os.path.isdir('./aug_data'):
        os.mkdir('./aug_data')
    if not os.path.isdir('./LLaVA/checkpoints'):
        os.mkdir('./LLaVA/checkpoints')

    if args.task_name == 'aug':
        if args.aug_mode == 'location':
            location_augment(args)
        elif args.aug_mode == 'action':
            action_augment(args)
        gen_instruct_format(args.output_aug_data, args.instruction_file)
    elif args.task_name == 'finetune':
        if args.finetune_both:
            if not os.path.exists(args.instruction_file):
                combine_both_aug_data(args.instruction_file, args.output_aug_images, args.output_aug_data)
        image_dir = args.output_aug_images
        input_file = args.instruction_file
        model_name = args.finetune_model_name
        output_dir = args.finetune_output_dir
        command = './LLaVA/scripts/finetune_aug.sh '+input_file+' '+image_dir+' '+model_name+' '+output_dir
        exit_code = subprocess.call([command], shell=True)
        print('finish finetuning for llava')
    
        merge_model_dir = args.merged_model_dir
        command_merge = './LLaVA/scripts/merge.sh '+output_dir+' '+model_name+' '+merge_model_dir
        exit_code = subprocess.call([command_merge], shell=True)
        print('finish merge the lora weight')            

    elif args.task_name == 'eval':
        
        if args.is_finetuned:
            if args.task_type == 'image':
                merge_model_dir = args.merged_model_dir
                eval_data = args.eval_data
                image_folder = './data/image/' 
                output_eval = args.output_eval
                command_eval = './LLaVA/llava/eval/gen_response.sh '+merge_model_dir+' '+eval_data+' '+image_folder+' '+output_eval
                exit_code = subprocess.call([command_eval], shell=True)
            elif args.task_type == 'video':
                pass 

            result = scoring(args.output_eval, args.matching_mode, args.label_mode)
            with open(args.output_score,'w') as fw:
                json.dump({args.matching_mode:result,
                           'eval_data':args.eval_data,
                           'output_eval':args.output_eval}, fw, indent=4, ensure_ascii=False)
        else:
            evaluate(args)

if __name__ == "__main__":
    main()

