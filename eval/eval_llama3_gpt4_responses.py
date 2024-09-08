# %%
import os
import pandas as pd
from tqdm import tqdm
import pickle
from eval_prompts import *
from openai import OpenAI
import time

# %% [markdown]
# Getting the responses for LLaMA-3

# %%
with open('llama3_response.pkl', 'rb') as file:
    llama3_responses = pickle.load(file)

llama3_responses.head()

# %% [markdown]
# Getting the responses from GPT-4

# %%
with open('gpt4_response.pkl', 'rb') as file:
    gpt4_responses = pickle.load(file)

gpt4_responses.head()

# %% [markdown]
# Merging the responses based on question

# %%
merged_responses = pd.merge(left=llama3_responses, right=gpt4_responses, how='left', on='counsel_chat_question')
merged_responses.head()

# %% [markdown]
# ### Setting up the GPT API

# %%
# Reading the OpenAI Key
with open('../api.key', 'r') as file:
    openai_api_key = file.read()
    
# Creating the client
client = OpenAI(api_key=openai_api_key)

# Setting up the chat format
def setup_chat_prompt(system_prompt, user_prompt, user_input, llama3_response, gpt4_response):
    user_prompt = user_prompt.format(user_input=user_input, llama3_response=llama3_response, gpt4_response=gpt4_response)
    
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    return messages

# Getting the openai response
def get_openai_response(messages, model="gpt-4", temperature=0, max_tokens=5000, n=1, stop=None, cnt=5):
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
            return get_openai_response(messages, model="gpt-4", temperature=0, max_tokens=5000, n=1, stop=None, cnt=cnt-1)
        
        print("bad text is ", messages)
        raise Exception("GPT Error")
    
def prompt_gpt(system_prompt, user_prompt, user_input, llama3_response, gpt4_response):
    
    messages = setup_chat_prompt(system_prompt=system_prompt, user_prompt=user_prompt, user_input=user_input, llama3_response=llama3_response, gpt4_response=gpt4_response)
    final_response = get_openai_response(messages = messages)
    
    return final_response

# %% [markdown]
# ### Getting the Evaluation Responses

# %%
eval_response = []

for index, row in tqdm(merged_responses.iterrows(), total=merged_responses.shape[0]):
    user_input = row['counsel_chat_question']
    llama3_response = row['llama3_response']
    gpt4_response = row['gpt4_response']
    
    response = prompt_gpt(system_rank_prompt, user_rank_prompt, user_input, llama3_response, gpt4_response)
    eval_response.append(response)

# %% [markdown]
# ### Storing the results

# %%
merged_responses['eval_llama3_gpt4_response'] = eval_response

with open('llama3_gpt4_evaluation.pkl', 'wb') as file:
    pickle.dump(merged_responses, file)

# %%



