import json
import pandas as pd
from similarity import scoring
from tqdm import tqdm
from gpt import openai_gpt_model
import subprocess

def get_prompt(request):
    
    prompt = """
    Here is an ambiguous request from a person
    '''
    {request}
    '''
    According to the ambiguous request and the image of the background,
    Please respond with a reflected action
    ## Output
    """

    return prompt.format(request=request)

def generate_response(model, f_name):

    df = pd.read_json(f_name, lines=True)
    assert len(df) == 400
    all_response = []
    for i, row in df.iterrows():
        print(i)
        text = row['text']
        image_f = row['image']
        question_id = row['question_id']
        prompt = get_prompt(text)
        image_path = './data/image/'+image_f
        reflection = model.generate(prompt, True, image_path) 
        all_response.append({
                'question_id':question_id,
                'prompt':prompt,
                'text':reflection,
                'model_id':model.get_model_name()
            })

    return all_response

def evaluate(args):
    if 'gpt' in args.eval_model:
        model = openai_gpt_model(args.eval_model)
        f_eval = args.eval_data
        reponses = generate_response(model, f_eval)
        with open(args.output_eval, 'w') as outfile1:
            for one_response in reponses:
                json.dump(one_response, outfile1)
                outfile1.write('\n')
    else:
        if args.task_type == 'image':
            model_dir = args.eval_model
            eval_data = args.eval_data
            image_folder = './data/image/' 
            output_eval = args.output_eval
            command_eval = './LLaVA/llava/eval/gen_response.sh '+model_dir+' '+eval_data+' '+image_folder+' '+output_eval
            exit_code = subprocess.call([command_eval], shell=True)
        elif args.task_type == 'video':
            model_dir = args.eval_model
            eval_data = args.eval_data
            video_folder = './data/video/' 
            output_eval = args.output_eval
            command_eval = './LLaVA-NeXT/gen_response.sh '+model_dir+' '+eval_data+' '+video_folder+' '+output_eval
            exit_code = subprocess.call([command_eval], shell=True)

        
    print('finish generating response with model ', args.eval_model)

    
    print('start scoring')
    result = scoring(args.output_eval, args.matching_mode, args.label_mode)
    # evaluation
    with open(args.output_score,'w') as fw:
        json.dump({args.matching_mode:result,
                   'eval_data':args.eval_data,
                   'output_eval':args.output_eval}, fw, indent=4, ensure_ascii=False)
    
    print('finish evaluation')
