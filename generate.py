import json
import pickle
import argparse
import csv
import time
from tqdm import tqdm
from template import LOCATION_3, REQUEST_2
import random
from gpt import openai_gpt_model
from diffusion import diffusion_model
from blip_diffusion import blipdiffusion_model

location_template = [LOCATION_3]


def generate_scenarios_by_location(model, num, location):
    
    total_dialogus = []
    for template in location_template:
        prompt = template.format(num=num, location=location)
        dialogues = model.generate(prompt, False, None)
        aug_dialogues = json.loads(dialogues)
        keys = aug_dialogues.keys()
        key = list(keys)[0]
        total_dialogus += aug_dialogues[key]
    
    return total_dialogus


def generate_location_dataset(args, location_file, llm_model_name, vision_model_name):
    
    llm_model = openai_gpt_model(llm_model_name)
    print('finish initialize model :', llm_model.get_model_name())
    img_model = diffusion_model(vision_model_name)
    print('finish initialize model :', img_model.get_model_name())

    with open(location_file, 'r', encoding='utf-8') as f:
        location_data = json.load(f)
    print('how many locations: ', len(location_data))

    aug_dialogues = []
    aug_times = args.aug_times
    aug_num = args.aug_num
    for t in range(aug_times):
        for location in tqdm(location_data):
            print(location+' '+str(t))            
            dialogues = generate_scenarios_by_location(llm_model, aug_num, location)
            for i, dialogue in enumerate(dialogues):
                try:
                    utterance = dialogue['Person A']
                    response = dialogue['Person B']
                    description = dialogue['BackgroundObject']
                except Exception as e:
                    print(e)
                    continue
                img_prompt = img_model.image_prompt.format(location=location, description=description)
                img_model.generate_image(img_prompt, location+'_'+str(i)+'_'+str(t), args.output_aug_images)
                
                response = response.replace('Person B','I')
                response = response.replace('Person A','you')
                aug_dialogues.append({'_id':location+'_'+str(i)+'_'+str(t), 
                                      'utterance':utterance, 
                                      'response':response, 
                                      'description':description, 
                                      'location':location})
            
            
            
        
    print('sample amount : ',len(aug_dialogues)) 

    
    return aug_dialogues

def generate_scenarios_by_action(model, num, action):
    prompt = REQUEST_2.format(num=num, reflected_action=action)
    dialogues = model.generate(prompt, False, None)
    aug_dialogues = json.loads(dialogues)
    keys = aug_dialogues.keys()
    key = list(keys)[0]
    return aug_dialogues[key]
    
def generate_action_dataset(args, action_file, llm_model_name, vision_model_name):

    llm_model = openai_gpt_model(llm_model_name)
    print('finish initialize model :', llm_model.get_model_name())
    blip_img_model = blipdiffusion_model(vision_model_name)
    print('finish initialize model :', blip_img_model.get_model_name())
    
    with open(action_file,'rb') as fr:
        label_class = pickle.load(fr)
    count = 0 
    print('how many actions : ', len(label_class))

    aug_dialogues = []
    aug_times = args.aug_times
    aug_num = args.aug_num
    for action in label_class:
        print(action)
        dialogues = generate_scenarios_by_action(llm_model, aug_num, action)    
        for i, dialogue in enumerate(dialogues):
            utterance = dialogue['Person A']
            response = dialogue['Person B']
            description = dialogue['BackgroundObject']
            
            #print('draw picture')            
            blip_img_model.generate_blip_image(str(count)+'_'+action, 'room', description, args.output_aug_images)
            #generate_image('photo, first-person perspective, room, '+description, str(count)+'_v5')
            aug_dialogues.append({'_id':str(count)+'_'+action,
                                  'utterance':utterance, 
                                  'response':response, 
                                  'description':description})
            count += 1

    print(len(aug_dialogues))
    print(count)
    return aug_dialogues


def location_augment(args):
    new_data = generate_location_dataset(args, args.location_file, args.llm_model_name, args.vision_model_name) #'./arta_corpus/data/location.json'
    
    with open(args.output_aug_data, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print('location augmentation finished')

def action_augment(args):
    new_data = generate_action_dataset(args, args.action_file, args.llm_model_name, args.vision_model_name)

    with open(args.output_aug_data, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print('action augmentation finished')

