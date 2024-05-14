import os
import re
from tasks.base import Task, DATA_PATH
from prompts.mentalhealth import *
from models import gpt
from datasets import load_dataset
import pandas as pd
from post_analysis_module import *
import json

class MentalHealthTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='sdoh_prompt_data.json'):
        """
        file: a json file with user input and SDoH prompt
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'mentalhealth', file)
        self.user_input = []
        self.sdoh_prompt = []
        with open(path, 'r') as f:
            user_sdoh_data = json.load(f)
        
        for data in user_sdoh_data:
            self.user_input.append(data['txt'])
            self.sdoh_prompt.append(data['sdoh_prompt'])
        
        self.steps = 2
        self.stops = ['\nCounsel:\n', None]
        # Loading dataset for counsel chat
        self.mental_health_data = load_dataset("nbertagnolli/counsel-chat")

    def __len__(self) -> int:
        return len(self.user_input)
    
    def get_input(self, idx: int) -> str:
        return self.user_input[idx]
    
    def get_cot_prompt_data(self, input : str) -> tuple:
        # Reading user input and generating the matching occurences of user inputs and counselling advice
        res = process_text(list(input), k=1)
        
        # Getting the prompt of determinant
        det_prompt = res[0]['prompt']
        
        # Getting the counselling advice
        counsel_advice = res[0]['matchedDocs'][0][2][0][1]

        # Getting the user input
        user_input = res[0]['matchedDocs'][0][1]
        
        # Cleaning up the counsel advice
        counsel_advice = counsel_advice.replace(u'\xa0', ' ')
        counsel_advice = counsel_advice.replace(u'\n', ' ')
        counsel_advice = counsel_advice.replace(u'  ', ' ')
        
        # Cleaning up the user input
        user_input = user_input.replace(u'\n', ' ')
        user_input = user_input.replace(u'  ', ' ')
        
        # Formatting the example
        example_format = '''User Input: {}\nCounsel: {}'''.format(user_input, counsel_advice)
        
        return (det_prompt, example_format)        

    def get_cot_sdoh_prompt_data(self, input : str, idx : int) -> tuple:
        # Reading user input and generating the matching occurences of user inputs and counselling advice
        res = process_text(list(input), k=1)
        
        # Getting the prompt of determinant
        det_prompt = res[0]['prompt']
        
        # Getting the counselling advice
        counsel_advice = res[0]['matchedDocs'][0][2][0][1]

        # Getting the user input
        user_input = res[0]['matchedDocs'][0][1]
        
        # Cleaning up the counsel advice
        counsel_advice = counsel_advice.replace(u'\xa0', ' ')
        counsel_advice = counsel_advice.replace(u'\n', ' ')
        counsel_advice = counsel_advice.replace(u'  ', ' ')
        
        # Cleaning up the user input
        user_input = user_input.replace(u'\n', ' ')
        user_input = user_input.replace(u'  ', ' ')
        
        # Getting the SDOH prompt from idx
        sdoh_prompt = self.sdoh_prompt[idx]
        
        # Formatting the example
        example_format = '''User Input: {}\nCounsel: {}'''.format(user_input, counsel_advice)
        
        return (det_prompt, example_format, sdoh_prompt)        


    def test_output(self, idx: int, output: str):
        output = output.split('Counsel:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*suitablity score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    def cot_prompt_dynamic_wrap(self, x: str, y:str='') -> str:
        determinant_prompt, example_format = self.get_cot_prompt_data(input=x)
        return cot_prompt_dynamic.format(det_prompt=determinant_prompt, example=example_format, input=x) + y

    def cot_sdoh_prompt_dynamic_wrap(self, x: str, y:str, idx:int) -> str:
        determinant_prompt, example_format, sdoh_prompt = self.get_cot_sdoh_prompt_data(input=x, idx=idx)
        return cot_sdoh_prompt_dynamic.format(det_prompt=determinant_prompt, example=example_format, input=x, sdoh=sdoh_prompt) + y

    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y
    
    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt.format(input=x)
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Counsel:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Counsel 1:\n{ys[0]}\n\Counsel 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more suitable counsel is 1' in compare_output:
            return 0
        elif 'more suitable counsel is 2' in compare_output:
            return 1
        elif 'two counsel are similarly suitable' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1