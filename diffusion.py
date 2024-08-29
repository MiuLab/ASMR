import torch
import transformers
from diffusers import DiffusionPipeline
import json


#IMAGE = 'photo, first-person perspective, woman, {location}, {description}'

class diffusion_model:
    def __init__(self, model_name):
        self.model = model_name
        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe.to("cuda")
        self.image_prompt = 'photo, first-person perspective, woman, {location}, {description}'
        
    def get_model_name(self):
        return self.model


    def generate_image(self, prompt, file_name, image_path):


        images = self.pipe(prompt=prompt).images[0]
        images.save(image_path+'/' + file_name+'.jpg')
        
        return None
