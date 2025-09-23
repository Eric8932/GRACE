# The code is modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

from .prompts.log_prompt_templates import *
from .prompts.gradient_descent_prompts import example_template,optimize_prompt_template_single_1st,optimize_prompt_template_single_2nd_simp
from ..test_helper import eval_instruction_with_loader
from ..utils import *
import re
import numpy as np


class GradientDescent():
    def __init__(self, 
                 task, 
                 base_model, 
                 optim_model,
                 print_log = True,
                 logger = None,
                 num_new_prompts = 1,
                 eval_dataloader=None):

        self.task = task
        self.base_model = base_model
        self.optim_model = optim_model
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts
        self.eval_dataloader = eval_dataloader

        self.optimize_prompt_tempelate_1st = optimize_prompt_template_single_1st

        self.example_template = example_template
        
        self._build_forward_prompts_func = task.build_forward_prompts_completion
        self._batch_forward_func = self.base_model.batch_forward_func
        

    def forward(self, batch, cur_prompt):
        batch_size = len(batch['question'])
        batch_prompts =self._build_forward_prompts_func(batch['question'], cur_prompt)
        responses = self._batch_forward_func(batch_prompts)
        
        for index,(p, r) in enumerate(zip(batch_prompts, responses)):
            self.logger.info(f"---------------\t\t{index}\t\t----------------")
            self.logger.info(f"Input:\n{p}")
            self.logger.info(f"Output:\n{r}")
        preds = self.task.batch_clean_responses(responses)
        
        labels = self.task.clean_labels(batch['answer'])
        correct = self.task.cal_correct(preds, labels)
        
        batch_logs = []
        for i in range(batch_size):
            batch_logs.append({
                'cur_prompt': cur_prompt,
                'question': batch['question'][i],
                'model_input': batch_prompts[i],
                'gt_answer':batch['answer'][i],
                'model_response': responses[i],
                'label':labels[i],
                'pred':preds[i],
                })
        if "ncbi" not in self.task.task_name.lower():
            metric = np.mean(correct)
        else:
            metric = self.task.cal_metric(preds, labels, batch['question'])
        forward_output = {
            'cur_prompt': cur_prompt,
            'correct':correct,
            'examples':batch_logs, 
            'acc':metric
            }
        
        if self.print_log:
            log_str = forward_log_tempelate.format(
                cur_prompt=cur_prompt,
                batch_prompts=batch_prompts,
                responses=responses,
                preds=preds,
                labels=labels,
                correct=forward_output['correct'],
                acc=forward_output['acc'])

            self.logger.info(log_str)
        return forward_output
    
    
    def _clean_self_eval_score(self, response):
        return re.findall(r'\d+', response)[-1]
    
    
    def _split_error_and_correct_examples(self, forward_output): 
        error_examples = []
        correct_examples = []
        wrong_count,correct_count = 0,0
        for i, example in enumerate(forward_output['examples']):
            if wrong_count>20:
                break
            if "format error" in example['pred']:
                continue
            if forward_output['correct'][i]==0:
                wrong_count += 1
                error_examples.append(self.example_template.format(
                    index=wrong_count+correct_count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            elif forward_output['correct'][i]==1:
                correct_count += 1
                correct_examples.append(self.example_template.format(
                    index=wrong_count+correct_count, 
                    question=example['model_input'],
                    label=example['label'], 
                    response=example['model_response'],
                    prediction=example['pred']))
            else:
                raise ValueError(f'_get_error_examples: invalid correct number {i} {forward_output}.')
        error_string = ''.join(error_examples)
        correct_string = ''.join(correct_examples)
        return error_string, correct_string
    

    def _clean_optim_response(self, optim_response):
        start_tags = ['<START>']
        end_tags = ['</START>', '<END>', '</END>']

        end_index = 1000000
        for start_tag in start_tags:
            start_index = optim_response.find(start_tag)
            if start_index!=-1:
                break
        start_tag ="" if start_index==-1 else start_tag
        start_index = 0 if start_index==-1 else start_index
       
        for end_tag in end_tags:
            end_index_temp = optim_response.find(end_tag, start_index)
            if end_index_temp !=-1 and end_index_temp<end_index:
                end_index = end_index_temp

        
        if end_index==100000:
            content = optim_response[start_index + len(start_tag):].strip()
        else:
            content = optim_response[start_index + len(start_tag):end_index].strip()
        return [content]



    def __call__(self):
        return None



    def optimize_wrong(self, cur_prompt,error_string,correct_string,optimize_prompt_tempelate):
        optimize_prompt = optimize_prompt_tempelate.format(
            cur_prompt=cur_prompt,
            error_string=error_string,
            correct_string=correct_string,
          )

        number_new_prompts = self.num_new_prompts
        response = self.optim_model.batch_forward_func([optimize_prompt]*number_new_prompts )
        optimized_prompt = [self._clean_optim_response(res)[0].strip() for res in response]
        if self.print_log:
            log_str = optimize_log_tempelate_1st.format(optimize_prompt=optimize_prompt,response=response)
            self.logger.info(log_str)
        return optimized_prompt
    
    def step_wrong(self, cur_prompt,forward_output):
        self.logger.info(f'cur_prompt: {cur_prompt}')
      
        error_string, correct_string = self._split_error_and_correct_examples(forward_output=forward_output)
        
        optimize_prompt_template =  optimize_prompt_template_single_1st
        optimized_prompts = self.optimize_wrong(
            cur_prompt=cur_prompt, 
            error_string=error_string,
            correct_string=correct_string,
            optimize_prompt_tempelate=optimize_prompt_template )
     
        
        
        return optimized_prompts
    

    def step_simp(self, cur_prompt):
        optimize_prompt = optimize_prompt_template_single_2nd_simp.format(
            cur_prompt=cur_prompt,
          )
        number_new_prompts = 1
        response = self.optim_model.batch_forward_func([optimize_prompt]*number_new_prompts )
        optimized_prompt = [self._clean_optim_response(res)[0].strip() for res in response]
        if self.print_log:
            log_str = optimize_log_tempelate_1st.format(optimize_prompt=optimize_prompt,response=response)
            self.logger.info(log_str)
        return optimized_prompt[0]