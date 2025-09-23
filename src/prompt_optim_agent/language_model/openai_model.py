import requests
import uuid
import time
import random
from openai import OpenAI

import global_vars



class OpenAIModel():
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        base_model : bool,
        **kwargs):
        self.temperature = temperature
        self.model = model_name
        self.base_model = base_model

        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


    def generate(self, input):
        assert isinstance(input, str)
    
        query = input.replace('"', '\\"') 
        sleep_time = 20
        max_retry = 5
        outputs = None

        for i in range(int(max_retry + 1)):
            if i > 0:
                print(
                    f"Generation: retry {i}/{max_retry} after sleeping for {sleep_time:.0f} seconds."
                )
                time.sleep(sleep_time)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": query},
                    ],
                    stream=False,
                    temperature=self.temperature,
                )
                outputs = response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
            if outputs:
                break
          
        if self.base_model :
            global_vars.base_api_count +=1
            global_vars.base_input_token +=response.usage.prompt_tokens
            global_vars.base_output_token+=response.usage.completion_tokens
        else:
            global_vars.target_api_count +=1
            global_vars.target_input_token +=response.usage.prompt_tokens
            global_vars.target_output_token+=response.usage.completion_tokens

        return outputs

    
    def batch_forward_func(self, batch_prompts):
        outputs = [0]*len(batch_prompts)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_index = {executor.submit(self.generate, batch_prompts[i]): i for i in range(len(batch_prompts))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except Exception as e:
                    outputs[index] = ""
                    print(f"SYSTEM_ERROR: {str(e)}")

        return outputs