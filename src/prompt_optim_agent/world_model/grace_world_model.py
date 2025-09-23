from .gradient_descent import *
from ..test_helper import eval_instruction_with_loader
from typing import Generic
from ..search_algo.base_algo import State, Action
from ..search_algo.grace_search import GraceNode

from collections import deque
import random
import numpy as np

class GraceSearchWorldModel():
    def __init__(
        self,
        task,
        logger,
        
        # model
        base_model: str,
        optim_model: str,
        iteration_num = 80,
        stop_early_thresh = 5,
        num_wrong_sample = 3,
        num_correct_sample = 3,
        num_new_prompts = 1,
        train_shuffle = True,
        train_batch_size: int = 8,
        test_batch_size: int = 200,
        eval_batch_size: int = 200,
        **kwargs
        ) -> None:
        
        self.task = task
        self.logger = logger
        self.base_model = base_model
        self.optim_model = optim_model

        self.iteration_num = iteration_num
        self.stop_early_thresh = stop_early_thresh
        self.num_correct_sample = num_correct_sample
        self.num_wrong_sample = num_wrong_sample

        self.train_dataloader = self.task.get_dataloader('train', 
                                                        batch_size=train_batch_size, 
                                                        shuffle=train_shuffle)
        self.train_data_iterator = self._infinite_data_loader(self.train_dataloader)
        self.buffer = deque()
        
        self.test_dataloader = self.task.get_dataloader('test', 
                                                        batch_size=test_batch_size, 
                                                        shuffle=False)
        self.eval_dataloader = self.task.get_dataloader('eval', 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=False)
        self.gradient_descent = GradientDescent(task=self.task, 
                                                logger=self.logger, 
                                                base_model=base_model, 
                                                optim_model=optim_model, 
                                                num_new_prompts = num_new_prompts,
                                                eval_dataloader = self.eval_dataloader)
    def _infinite_data_loader(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch
                
    def get_train_batch(self):
        return next(self.train_data_iterator)
    


    

    def sample_forward_output(self,forward_output, num_wrong=3, num_right=3):
        examples = forward_output['examples']

        if "ncbi" not in self.task.task_name.lower():
            valid_examples = [ex for ex in examples if "format error" not in ex['pred'].lower()]
        else:
            valid_examples = examples
    
        wrong_samples = [ex for ex in valid_examples if ex['label'] != ex['pred']]
        right_samples = [ex for ex in valid_examples if ex['label'] == ex['pred']]

        selected_wrong = random.sample(wrong_samples, min(num_wrong, len(wrong_samples)))
        selected_right = random.sample(right_samples, min(num_right, len(right_samples)))
        
        selected = selected_right+selected_wrong

        new_forward_output = {
            'cur_prompt': forward_output['cur_prompt'],
            'examples': selected,
            'correct': [int(ex['label'] == ex['pred']) for ex in selected],
            'acc': np.mean([ex['label'] == ex['pred'] for ex in selected])
        }
        return new_forward_output
    


    def check_number(self,forward_output):
        examples = forward_output['examples']

        if "ncbi" not in self.task.task_name.lower():
            valid_examples = [ex for ex in examples if "format error" not in ex['pred'].lower()]
        else:
            valid_examples = examples
    
        wrong_samples = [ex for ex in valid_examples if ex['label'] != ex['pred']]
        right_samples = [ex for ex in valid_examples if ex['label'] == ex['pred']]
        if len(right_samples)>=self.num_correct_sample and len(wrong_samples)>=self.num_wrong_sample:
            return True
        return False
        
    def train_forward(self,cur_prompt):
        aggregated_output = {
            'cur_prompt': cur_prompt,
            'correct': [],
            'examples': [],
            'acc': []
        }
        
        for batch in self.train_dataloader:
            forward_output_cur = self.gradient_descent.forward(batch=batch, cur_prompt=cur_prompt)
            aggregated_output['correct'] += forward_output_cur['correct']
            aggregated_output['examples'] += forward_output_cur['examples']
            aggregated_output['acc'].append(forward_output_cur['acc'])
            if self.check_number(aggregated_output):
                break

        aggregated_output['acc'] = np.mean(aggregated_output['acc'])

        return aggregated_output
        

    def _sort_helper(self, metric):
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric
        
    def _gradient_descent_step(self, node: GraceNode):

        new_nodes = []
        child_node = node
       
        #Get train batch
        # train_batch = {"question":[],'answer':[]}
        # for batch in self.train_dataloader:
        #     train_batch['question']+=batch['question']
        #     train_batch['answer']+=batch['answer']

        #Get eval batch
        eval_batch = {"question":[],'answer':[]}
        for batch in self.eval_dataloader:
            eval_batch['question']+=batch['question']
            eval_batch['answer']+=batch['answer']


        #Get initial eval score
        eval_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=child_node.prompt)
        child_node.eval_metric = eval_forward_output['acc']
        max_acc = self._sort_helper(eval_forward_output['acc'])

        #Split train into correct and wrong samples
        
        stop_early = 0

        for iter in range(self.iteration_num):
            cur_child_node = child_node
            self.logger.info(f'------------------  expand node {cur_child_node.id} ---------------------')

            train_forward_output = self.train_forward(cur_prompt=child_node.prompt)
            cur_acc = self._sort_helper(train_forward_output['acc'])
            if int(cur_acc)==1:
                break
           
            sampled_forward_output = self.sample_forward_output(train_forward_output,self.num_wrong_sample,self.num_correct_sample)


            self.logger.info(f'----------------  OPTIMIZATION batch {iter} ----------------')
            optimized_prompts = self.gradient_descent.step_wrong(child_node.prompt, forward_output = sampled_forward_output)
            for opt_prompt in optimized_prompts:
                eval_temp_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=opt_prompt)
                temp_child_node = GraceNode(
                    prompt=opt_prompt, 
                    action="OPT",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'],eval_temp_forward_output['acc'])
                new_nodes.append(temp_child_node)
                
                #Change current prompt
                if self._sort_helper(eval_temp_forward_output['acc'])>max_acc:
                    stop_early = 0
                    max_acc = self._sort_helper(eval_temp_forward_output['acc'])
                    child_node = temp_child_node
                    #Re-Split train into correct and wrong samples
                    # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
                    eval_forward_output = eval_temp_forward_output
                else:
                    stop_early+=1

            if stop_early==self.stop_early_thresh:
                self.logger.info(f'----------------  SIMPLIFY batch {iter} ----------------')
                simp_prompt = self.gradient_descent.step_simp(child_node.prompt)
                eval_temp_forward_output = self.gradient_descent.forward(batch=eval_batch, cur_prompt=simp_prompt)
                temp_child_node = GraceNode(
                    prompt=simp_prompt, 
                    action="SIMP",
                    mom_prompt=None,
                    parent=cur_child_node,
                    )
                temp_child_node.eval_metric = (eval_forward_output['acc'],eval_temp_forward_output['acc'])
                new_nodes.append(temp_child_node)
                
                stop_early = 0
                max_acc = self._sort_helper(eval_temp_forward_output['acc'])
                child_node = temp_child_node
                # train_forward_output = self.train_forward(cur_prompt=opt_prompt)
                eval_forward_output = eval_temp_forward_output

        return new_nodes, None


    
    def step(self, node:GraceNode):
        new_nodes, gradient_descent_output = self._gradient_descent_step(node=node)
        return new_nodes, gradient_descent_output
    
    def build_root(self, init_prompt):
        node = GraceNode(prompt=init_prompt, action=None, parent=None,mom_prompt=None)

        return node
    

    def test_prompt(self, prompt):
        metric, eval_output = eval_instruction_with_loader(task=self.task, 
                                           eval_prompt=prompt,
                                           dataloader=self.test_dataloader,
                                           base_model=self.base_model,
                                           )
        return metric, eval_output
    


