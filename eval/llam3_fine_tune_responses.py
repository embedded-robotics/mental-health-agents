# %%
import os
import pandas as pd
from tqdm import tqdm
import pickle

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer import TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_int8_training, PeftModel
from trl import SFTTrainer, DPOTrainer
from datasets import load_dataset, Dataset
from transformers import pipeline
import numpy as np

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
# ### Loading the merged model for inference

# %%
model = AutoModelForCausalLM.from_pretrained(
    "../src/dpo/llama-3-8b-glan-dpo-merged",
    torch_dtype = torch.bfloat16,
    device_map={'':torch.cuda.current_device()}
)

# %%
tokenizer = AutoTokenizer.from_pretrained("../src/dpo/llama-3-8b-glan-dpo-merged")

# %% [markdown]
# ### Mapping the counsel chat questions to Chat Template

# %%
def format_chat_template_questions(row):
    row_json = [
        {"role" : "user", "content": row['question']}
    ]

    prompt = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=True)

    return {
        "prompt": prompt
    }

# %%
dataset = counsel_dataset.map(
                format_chat_template_questions,
                num_proc=24
                )

# %%
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# %%
model.config.use_cache = True

# %%
prompt_questions_list = []
prompt_output_list = []
BATCH_SIZE = 10

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    
    prompt_questions = dataset[i:i+BATCH_SIZE]['question']
    prompts = dataset[i:i+BATCH_SIZE]['prompt']
    
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(model.device)
    generation_config = model.generation_config
    generation_config.pad_token_id = tokenizer.pad_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        num_return_sequences=1,
        generation_config=generation_config
    )
    
    prompts_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    prompts_output = [output.split('assistant')[1].strip() for output in prompts_output]
    
    prompt_questions_list = prompt_questions_list + prompt_questions
    prompt_output_list = prompt_output_list + prompts_output

# %%
response_dict = {'counsel_chat_question': prompt_questions_list,
                 'llama3_response': prompt_output_list}

response_df = pd.DataFrame(response_dict)

with open('llama3_response.pkl', 'wb') as file:
    pickle.dump(response_df, file)

# %%



