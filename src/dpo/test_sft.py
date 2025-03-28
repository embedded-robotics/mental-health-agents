# %% [markdown]
# ### Fine-Tuning LLaMA-3.2 3B Instruct
# 
# This code will fine-tune the `LLaMA-3.2-3B-Instruct` model on the CounselChat dataset

# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import torch
import pickle
from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from datasets import Dataset, DatasetDict
from trl import SFTTrainer

# %%
os.getenv('LD_LIBRARY_PATH')

# %% [markdown]
# Some links to be used for fine-tuning
# 1. https://www.kdnuggets.com/fine-tuning-llama-using-unsloth
# 2. https://www.analyticsvidhya.com/blog/2024/12/fine-tuning-llama-3-2-3b-for-rag/
# 3. https://www.linkedin.com/pulse/step-guide-use-fine-tune-llama-32-dr-oualid-soula-xmnff/
# 4. https://medium.com/@alexandros_chariton/how-to-fine-tune-llama-3-2-instruct-on-your-own-data-a-detailed-guide-e5f522f397d7 (this seems faulty since it copies the output id's within the input_ids)
# 5. https://drlee.io/step-by-step-guide-fine-tuning-metas-llama-3-2-1b-model-f1262eda36c8
# 6. https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article
# 7. https://blog.futuresmart.ai/fine-tune-llama-32-vision-language-model-on-custom-datasets
# 8. https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/multigpu_finetuning.md
# 
# 

# %% [markdown]
# ### Reading the Counsel Chat Dataset

# %%
with open('processed_data/counselchat_top_votes_train.pkl', 'rb') as file:
    dataset_top_votes_train = pickle.load(file)

dataset_top_votes_train.head()

# %%
with open('processed_data/counselchat_top_votes_test.pkl', 'rb') as file:
    dataset_top_votes_test = pickle.load(file)

dataset_top_votes_test.head()

# %% [markdown]
# Creating the HuggingFace dataset using Pandas Dataframe

# %%
dataset_train = Dataset.from_pandas(dataset_top_votes_train)
dataset_test = Dataset.from_pandas(dataset_top_votes_test)

# %%
dataset = DatasetDict()
dataset['train'] = dataset_train
dataset['test'] = dataset_test

# %%
dataset

# %% [markdown]
# ### Fine-Tuning Code

# %% [markdown]
# #### Loading the model and tokenizer

# %%
max_seq_length = 2048 
dtype = None # None for auto-detection.
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=dtype,
    device_map="auto"
)

# %% [markdown]
# #### Forming the chat template

# %%
# Define a function to apply the chat template
def format_chat_template(example):
        
    messages = [
        {"role": "system", "content": "You are an expert mental health professional trained to counsel and guide patients suffering from ill mental-health"},
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answerText']}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text": prompt}

# %%
dataset_formatted = dataset.map(format_chat_template)

# %%
print(dataset_formatted['train']['text'][0])

# %% [markdown]
# #### Initializing the TRL SFTTrainer and related Arguments

# %%
new_model = "./llama32-sft-fine-tune-counselchat"

training_args = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="epoch",
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset=dataset_formatted["train"],
    eval_dataset=dataset_formatted["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args)

# %%
trainer.train_dataset

# %%
print(tokenizer.decode(trainer.train_dataset['input_ids'][0]))

# %% [markdown]
# #### Only Focus on the `Response Part` for the generation

# %%
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# %%
trainer.train_dataset

# %%
# The labels are created which only contain response. Left Padding is implemented and all the padding tokens are given a score of -100 to avoid loss calculation for pad_tokens
trainer.train_dataset['labels'][0]

# %% [markdown]
# #### Train the model

# %%
trainer_stats = trainer.train()

# %% [markdown]
# #### Saving the model and tokenizer

# %%
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# %% [markdown]
# ### Inference

# %%
FastLanguageModel.for_inference(model)

messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I don't have the will to live anymore. I don't feel happiness in anything. "}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text.split("assistant")[1])

# %%



