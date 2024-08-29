import os
import json
import csv
import time 
import openai

OPENAI_API_KEY = ''
openai.api_key = OPENAI_API_KEY


def translate(J_sentence):
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo-16k-0613',
        messages = [
            {
                'role':'system',
                'content':'You will be provided with a sentence in Japanese, and your task is to translate it into English.'
            },
            {
                'role':'user',
                'content':J_sentence
            }
        ],
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0

    )


    return response["choices"][0]['message']['content']

def translate_data(file_name):
    
    with open(file_name,'r',encoding='utf-8') as f:
        scenario_data = [json.loads(line) for line in f]
    
    en_scenario = []
    for sample in scenario_data:
        action = sample['action']
        utterance = sample['utterance']
        description = sample['description']
        
        e_action = translate(action)
        e_utterance = translate(utterance)
        e_description = translate(description)

        time.sleep(5)
        
        en_scenario.append({'filename':sample['filename'],'action':e_action,'utterance':e_utterance,'description':e_description})
        print(len(en_scenario))
        
    return en_scenario

def main():
    
    en_scenario_data = translate_data('./data/scenario/scenario.json')

    with open('./data/scenario/scenario_en.json','w',encoding='utf-8') as fw:
        json.dump(en_scenario_data, fw)

if __name__ == "__main__":
    main()
