import json
import random
import shutil
import os

    
def gen_instruct_format(output_aug_data, instruction_file):
    with open(output_aug_data,'r',encoding='utf-8') as f:
        all_samples = json.load(f)

    print(len(all_samples))
    answer = []
    for i, one_sample in enumerate(all_samples):

        utterance = one_sample['utterance']
        response = one_sample['response']
        file_name = one_sample['_id']
        

       
        answer.append({
            'id' : i, 
            'image': file_name+'.jpg',
            'conversations' : [
                {
                    'from':'human',
                    'value': '<image>\n'+utterance

                },
                {
                    'from':'gpt',
                    'value': response
                }

            ]        
        })

    print(len(answer))
    with open(instruction_file,'w') as out_f:

        json.dump(answer, out_f, ensure_ascii=False, indent=4)


def combine_both_aug_data(instruction_file, images_folder, both_data):

    with open('./aug_data/new_dataset_location.json','r',encoding='utf-8') as f1:
        location_samples = json.load(f1)
    with open('./aug_data/new_dataset_action.json','r',encoding='utf-8') as f2:
        action_samples = json.load(f2)

    total_samples = location_samples + action_samples
    random.shuffle(total_samples)
    with open(both_data,'w') as out_f:
        json.dump(total_samples, out_f, ensure_ascii=False, indent=4)

    os.mkdir(images_folder)
    for file in os.listdir('./aug_images_location/'):
        shutil.copy('./aug_images_location/'+file, images_folder)

    for file in os.listdir('./aug_images_action/'):
        shutil.copy('./aug_images_action/'+file, images_folder)

    gen_instruct_format(both_data, instruction_file)
