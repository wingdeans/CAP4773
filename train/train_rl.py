from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import textwrap

checkpoint = "./CodeLlama-7b-hf"

# Load, split, tokenize, collate dataset

dataset = load_dataset("json", data_files={"train": "train/train.json", "test": "train/test.json"})

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)

def preprocess(batch):
    batch = ((src, textwrap.indent(tgt, "\t")) for src, tgt in zip(batch["src"], batch["tgt"]))
    return tokenizer([f"<ASM>\n{src}</ASM>\nfunc main() {{\n{tgt}\n}}</s>" for src, tgt in batch])

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names
)

print(dataset)

dataset = dataset.filter(lambda x: len(x["input_ids"]) < 7000)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

print(dataset)

# Load model

from trl import AutoModelForCausalLMWithValueHead
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

model = LlamaForCausalLM.from_pretrained(
    checkpoint, device_map={"": 0}, load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="train/long",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
)

# class CustomTrainer(Trainer):
#     def training_step(self, *args, **kwargs):
#         print(args[1]["input_ids"].shape)
#         # print_gpu_utilization()
#         return super(CustomTrainer, self).training_step(*args, **kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train(resume_from_checkpoint=True)
