#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model
import torch

# -------------------------------------------------------------------------------------------------------------------------------------------
#  Converts dataset to Tokenized dataset and than creates two identical training sets: masked prompt + unmasked text / unmasket prompt + unmasked text 
# -------------------------------------------------------------------------------------------------------------------------------------------

# Tokenization
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def tokenize_dat(examples):
    tokenized = tokenizer(
        examples["full_text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    
    MAX_LEN = 512
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    #creating mask so prompt has tokenized value of -100 so is not included in answer
    prompt_len = min(examples["prompt_len"], MAX_LEN)
    labels[:prompt_len] = [-100] * prompt_len

    # masking padding tokens
    pad_id = tokenizer.pad_token_id
    labels = [-100 if tok == pad_id else lab for tok, lab in zip(input_ids, labels)]
    
    return {"input_ids": input_ids, "attention_mask": tokenized["attention_mask"], "labels": labels}


def format_chat(examples):
    messages = [
        {"role": "user", "content": examples["instruction"]},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        padding=False,
        max_length=512,
    )["input_ids"]

    prompt_len = len(prompt_ids)
    # creating template of prompt, retrieving id of each prompt and getting their lenght for the mask

    # full text (Prompt + answer)
    full_text = prompt_text + examples["response"]

    return {"full_text": full_text, "prompt_len": prompt_len}
    

def main ():

    validator = input("train Y/N?")
    if validator == "Y":
        dataset = load_dataset("json", data_files="historical_dataset.jsonl")["train"]
        dataset = dataset.map(format_chat) # <--- full_text, prompt_len
        dataset = dataset.map(tokenize_dat) # <--- input_ids, attention_mask, labels # passing text to tokenizer to create tokenized version with masked prompt
        dataset = dataset.remove_columns(["instruction", "response","full_text", "prompt_len"])# removing unnecessary data
        # Now dataset contains: input_ids, attention_mask, labels
        dataset = dataset.train_test_split(test_size=0.2)
    
    
        #training
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda")
    
        # LoRA configuration 
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
        # Training arg setup
        training_args = TrainingArguments(
        output_dir="./tinyllama-historicalV2",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        )
    
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
        )
    
        trainer.train()
        trainer.save_model("./tinyllama-historicalV2")
    
         # ===============================================
        # Test the Fine-Tuned Model
        # ===============================================
        from transformers import pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
        messages = [
            {"role": "user", "content": "What is the most common way of transport?"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ← tells it “now the assistant should answer”
        )
        
        result = pipe(prompt, max_new_tokens=100)
        print(result[0]["generated_text"])
    
    else:
        print("Done, not trained")

    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




