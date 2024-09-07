# %%
import os
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
from datasets import load_dataset, Dataset
from eval_prompts import *
from openai import OpenAI
import time

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# %% [markdown]
# ### Preparing the counsel chat dataset for evaluation

# %%
dataset = load_dataset("nbertagnolli/counsel-chat")
question_id, question_id_index = np.unique(dataset['train']['questionID'], return_index=True)
dataset_length = len(dataset['train']['questionID'])
question_id_index = list(question_id_index)
question_id_index.append(dataset_length)

# %%
questions = []
preferred_answers = []
rejected_answers = []

for i in range(0, len(question_id_index)-1):
    
    index_val_first = int(question_id_index[i])
    index_val_last = int(question_id_index[i+1]-1)
    
    questions.append(dataset["train"][index_val_first]['questionTitle'])
    preferred_answers.append(dataset["train"][index_val_first]['answerText'])
    rejected_answers.append(dataset["train"][index_val_last]['answerText'])
    

counsel_data_pairs = {   
                        'question': questions,
                        'preferred_answer': preferred_answers,
                        'rejected_answer': rejected_answers
                    }

counsel_dataset = Dataset.from_dict(counsel_data_pairs)

# %%
counsel_dataset

# %% [markdown]
# ### Setting up the API calls for GPT-4

# %%
# Reading the OpenAI Key
with open('../api.key', 'r') as file:
    openai_api_key = file.read()
    
# Creating the client
client = OpenAI(api_key=openai_api_key)

# Setting up the chat format
def setup_chat_prompt(system_prompt, user_prompt, user_input):
    user_prompt = user_prompt.format(user_input=user_input)
    
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    return messages

# Getting the openai response
def get_openai_response(messages, model="gpt-4", temperature=0.7, max_tokens=5000, n=1, stop=None, cnt=5):
    try:
        output = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop
        )
        
        return output.choices[0].message.content
    
    except Exception as E:
        print(E)
        time.sleep(3*(5-cnt))
        if cnt != 0:
            return get_openai_response(messages, model="gpt-4", temperature=0.7, max_tokens=5000, n=1, stop=None, cnt=cnt-1)
        
        print("bad text is ", messages)
        raise Exception("GPT Error")
    
def prompt_gpt(system_prompt, user_prompt, user_input):
    
    messages = setup_chat_prompt(system_prompt=system_prompt, user_prompt=user_prompt, user_input=user_input)
    final_response = get_openai_response(messages = messages)
    
    return final_response

# %%
prompt_questions_list = []
prompt_output_list = []

for i in tqdm(range(0, len(counsel_dataset))):    
    prompt_question = counsel_dataset[i]['question']
    prompts_output = prompt_gpt(system_prompt=system_cot_prompt, user_prompt=user_cot_prompt, user_input=prompt_question)
    
    prompt_questions_list.append(prompt_question)
    prompt_output_list.append(prompts_output)

# %%
response_dict = {'counsel_chat_question': prompt_questions_list,
                 'gpt4_response': prompt_output_list}

response_df = pd.DataFrame(response_dict)

with open('gpt4_response.pkl', 'wb') as file:
    pickle.dump(response_df, file)
