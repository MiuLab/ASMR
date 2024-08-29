from diffusers.pipelines import BlipDiffusionPipeline
import torch
from diffusers.utils import load_image
import json
import os
import random


class blipdiffusion_model:
    def __init__(self, model_name):
        self.model = model_name
        self.blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        self.images = os.listdir('./data/image/')
    
    def get_model_name(self):
        return self.model

    def generate_blip_image(self, image_name, subject, description, image_path):


        cond_subject = subject
        tgt_subject = subject

        text_prompt_input = 'photo, first-person perspective, '+description
        rand_image = random.choice(self.images)
        cond_image = load_image("./data/image/"+rand_image)

        guidance_scale = 7.5

        num_inference_steps = 50
        negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

        output = self.blip_diffusion_pipe(
            text_prompt_input,
            cond_image,
            cond_subject,
            tgt_subject,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=1024,
            width=1024,
        ).images


        output[0].save(image_path+'/'+image_name+'.jpg')



