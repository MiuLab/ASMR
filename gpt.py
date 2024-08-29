import os
import time
import base64
from openai import OpenAI

class openai_gpt_model:
    def __init__(self, model_name):
        self.client = OpenAI(api_key='')
        self.model = model_name
    
    def get_model_name(self):
        return self.model

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, content, contain_V, image_path):
        
        if contain_V:
            while(True):
                try:
                    base64_image = self.encode_image(image_path)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": content},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.8,
                        max_tokens=512,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0

                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            return response.choices[0].message.content

        else:
            while(True):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        response_format={ "type": "json_object" },
                        messages=[
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "You are a helpful assistant."
                                    }
                                ]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": content
                                    }
                                ]
                            }
                        ],
                        temperature=0.8,
                        max_tokens=2048,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            return response.choices[0].message.content
