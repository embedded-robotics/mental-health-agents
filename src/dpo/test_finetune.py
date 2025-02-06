# %%
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# %% [markdown]
# ### Preparing the Counsel Chat Dataset

# %%
dataset_name = "nbertagnolli/counsel-chat"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42)

# %% [markdown]
# ### Writing the Fine-Tuning Code

# %%
with open('hf_token.key', 'r') as f:
    hf_token = f.read()

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Adding a special token for pad token so that eos token can be recognized 
# (https://github.com/unslothai/unsloth/issues/416)
# https://github.com/huggingface/transformers/issues/22794
# https://github.com/huggingface/transformers/issues/23230
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

# %%
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto") # Must be float32 for MacBooks!

model.config.use_cache=False
model.config.pad_token_id = tokenizer.pad_token_id # Updating the model config to use the special pad token

# %%
# Define a function to apply the chat template
def format_chat_template(example):
    question_text_title_combined = None
    
    if example['questionTitle'] == None:
        question_text_title_combined = example['questionText']
    elif example['questionText'] == None:
        question_text_title_combined = example['questionTitle']
    else:
        question_text_title_combined = example['questionText'] + " " + example['questionTitle']
    
    messages = [
        {"role": "user", "content": question_text_title_combined},
        {"role": "assistant", "content": example['answerText']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

# %%
new_dataset = dataset.map(format_chat_template)
new_dataset = new_dataset.train_test_split(0.05)

# %%
# Tokenize the data
def tokenize_function(example):
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True)
    # Set padding token labels to -100 to ignore them in loss calculation
    tokens['labels'] = [
        -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
    ]
    return tokens

# %%
# Apply tokenize_function to each row
tokenized_dataset = new_dataset.map(tokenize_function)
tokenized_dataset = tokenized_dataset.remove_columns(['questionID', 'questionTitle', 'questionText', 'questionLink', 'topic', 'therapistInfo', 'therapistURL', 'answerText', 'upvotes', 'views', 'prompt'])

# %%
model.train()
training_args = TrainingArguments(
    output_dir="./llama32-sft-fine-tune-counselchat",
    eval_strategy="steps", # To evaluate during training
    eval_steps=50,
    logging_steps=50,
    save_steps=500,
    per_device_train_batch_size=2, # Adjust based on your hardware
    per_device_eval_batch_size=2,
    num_train_epochs=2, # How many times to loop through the dataset
    fp16=False, # Must be False for MacBooks
    report_to="none", # Here we can use something like tensorboard to see the training metrics
    log_level="info",
    learning_rate=1e-5, # Would avoid larger values here
    max_grad_norm=2 # Clipping the gradients is always a good idea
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer)

# %%
# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./llama32-sft-fine-tune-counselchat")
tokenizer.save_pretrained("./llama32-sft-fine-tune-counselchat")

# %%



