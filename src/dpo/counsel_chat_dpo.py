import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer import TrainingArguments
from tqdm import tqdm
from peft import LoraConfig, TaskType, AutoPeftModelForCausalLM
from trl.trainer import ConstantLengthDataset
from trl import SFTTrainer, DPOTrainer

model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("nbertagnolli/counsel-chat")
question_id, question_id_index = np.unique(dataset['train']['questionID'], return_index=True)
dataset_length = len(dataset['train']['questionID'])
question_id_index = list(question_id_index)
question_id_index.append(dataset_length)

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

counsel_dataset = counsel_dataset.train_test_split(test_size=0.1, seed=42)
train_data = counsel_dataset['train']
test_data = counsel_dataset['test']

def prepare_sample_text(example):
    text = f"Question: {example['question']}\n\nCounsel Advice: {example['preferred_answer']}"
    return text

def chars_token_ratio(dataset, tokenizer):
    '''
    Estimate the average number of characters per token in the dataset
    '''
    
    total_characters, total_tokens = 0, 0
    dataset_length = len(dataset['question'])
    for _, example in tqdm(zip(range(dataset_length), iter(dataset)), total=dataset_length):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    
    return total_characters/total_tokens

chars_per_token = chars_token_ratio(train_data, tokenizer)

train_dataset = ConstantLengthDataset(
    tokenizer,
    train_data,
    formatting_func=prepare_sample_text,
    infinite=True,
    seq_length=1024,
    chars_per_token=chars_per_token
)

test_dataset = ConstantLengthDataset(
    tokenizer,
    test_data,
    formatting_func=prepare_sample_text,
    infinite=False,
    seq_length=1024,
    chars_per_token=chars_per_token
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = "auto",
    # device_map = {"":0},
    torch_dtype = torch.bfloat16,
    trust_remote_code = False
)

base_model.config.use_cache=False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type = TaskType.CAUSAL_LM
)

training_args=TrainingArguments(
    output_dir="counsel_data_sft",
    num_train_epochs=5,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps = 25,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    group_by_length=False,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.05,
    optim="paged_adamw_32bit",
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

sft_trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        packing=True,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args
    )

sft_trainer.train()