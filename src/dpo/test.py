# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

# %% [markdown]
# ### Get the CounselChat Dataset

# %%
dataset_name = "nbertagnolli/counsel-chat"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42)

# %%
dataset_df = dataset.to_pandas()
dataset_df.head()

# %%
dataset_df_top_votes = dataset_df.groupby('questionID').apply(lambda x: x.sort_values('upvotes', ascending=False).iloc[0], include_groups=False).reset_index()
dataset_df_top_votes

# %%
dataset_df_top_votes['question'] = dataset_df_top_votes['questionText'] + " " + dataset_df_top_votes['questionTitle']
dataset_df_top_votes

# %%
dataset_df_final = dataset_df_top_votes[['topic', 'question', 'answerText']]
dataset_df_final

# dataset_df_final = dataset_df_final.dropna().reset_index(drop=True)

# def remove_emojis(df_bios):
#     emoj = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002500-\U00002BEF"  # chinese char
#         u"\U00002702-\U000027B0"
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u"\U00010000-\U0010ffff"
#         u"\u2640-\u2642" 
#         u"\u2600-\u2B55"
#         u"\u200d"
#         u"\u23cf"
#         u"\u23e9"
#         u"\u231a"
#         u"\ufe0f"  # dingbats
#         u"\u3030"
#                       "]+", re.UNICODE)
#     return re.sub(emoj, '', df_bios)

# dataset_df_final['question'] = dataset_df_final['question'].apply(remove_emojis)
# dataset_df_final['question'] = dataset_df_final['question'].apply(lambda x: x.replace('\xa0', ' ').replace(u"\u2019","'").replace(u"\u00e9","ee").replace("\n",' ').replace("  ", " "))

# dataset_df_final['answerText'] = dataset_df_final['answerText'].apply(remove_emojis)
# dataset_df_final['answerText'] = dataset_df_final['answerText'].apply(lambda x: x.replace('\xa0', ' ').replace(u"\u2019","'").replace(u"\u00e9","ee").replace("\n",' ').replace("  ", " "))

# %% [markdown]
# ### OpenAI Configuration and Responses

# %%
with open("../../api.key", 'r') as file:
    openai_api_key = file.read()

openai_client = OpenAI(api_key=openai_api_key)

# %% [markdown]
# ### OpenAI Fine-Tuned Model Response

# %%
def get_openai_response_finetuned(system_prompt: str, user_prompt: str) -> str:
        
    completion = openai_client.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:university-of-texas-at-austin:counselchat-clean:BE3PqwuO",
    temperature=0,
    max_tokens=2048,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
    )

    openai_response = completion.choices[0].message.content
    
    return openai_response

system_prompt_qa = 'You are an expert mental health professional trained to counsel and guide patients suffering from ill mental-health'

# %%
gpt_responses_ft = []
for index, row in tqdm(dataset_df_final.iterrows(), total=len(dataset_df_final)):
    question_input = row['question']
    try:
        gpt_resp = get_openai_response_finetuned(system_prompt=system_prompt_qa, user_prompt=question_input)
        gpt_responses_ft.append(gpt_resp)
    except:
        gpt_responses_ft.append('')
        
with open('test_inference_data/openai_ft_que_resp.pkl', 'wb') as file:
    pickle.dump(gpt_responses_ft, file)