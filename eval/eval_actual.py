from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import textwrap

checkpoint = "./CodeLlama-7b-hf"
adapter = "./train/long/checkpoint-9001"

# Load, split, tokenize, collate dataset

dataset = load_dataset("json", data_files={"test": "train/test.json"})

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)

def preprocess(batch):
    return tokenizer([f"<ASM>\n{src}</ASM>\nfunc main() {{\n" for src in batch["src"]])

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=dataset["test"].column_names
)

dataset = dataset.filter(lambda x: len(x["input_ids"]) < 7000) # REMOVEME

tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

print(dataset)

# Load model

from transformers import LlamaForCausalLM, TrainingArguments, Trainer
from peft import PeftModel

model = LlamaForCausalLM.from_pretrained(
    checkpoint, device_map={"": 0}, load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = PeftModel.from_pretrained(model, adapter)

def compute_metrics(eval_pred):
    print(eval_pred)
    return {"deez": "nuts"}

training_args = TrainingArguments(
    output_dir="train/long",
    evaluation_strategy="epoch",
#   save_strategy="epoch",
#   num_train_epochs=15,
#   learning_rate=2e-5,
#   weight_decay=0.01,
#   per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=16,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    # train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# with torch.no_grad():
trainer.evaluate()
