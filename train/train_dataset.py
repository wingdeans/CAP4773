# import gc
import os
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()

from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

checkpoint = "./CodeLlama-7b-hf"

# Load, split, tokenize, collate dataset

print_gpu_utilization()

dataset = load_dataset("json", data_files="train/test.json")
dataset = dataset["train"].train_test_split(test_size=0.2)

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)

def preprocess(pairs):
    # return tokenizer([f"<ASM>\n{pairs['src']}</ASM>\n{pairs['tgt']}"
    return tokenizer([f"<ASM>\n{src}</ASM>\n{tgt}"
                      for src, tgt in zip(pairs["src"], pairs["tgt"])])

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=dataset["train"].column_names
)

block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

print(dataset)
print_gpu_utilization()

# Load model

from transformers import LlamaForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

model = LlamaForCausalLM.from_pretrained(
    checkpoint, device_map={"": 0}, load_in_4bit=True,
    # checkpoint, device_map="auto", load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print_gpu_utilization()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8, lora_alpha=32,
    lora_dropout=0.1
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print_gpu_utilization()

training_args = TrainingArguments(
    output_dir="train/out",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
)

# class CustomTrainer(Trainer):
#     def training_step(self, *args, **kwargs):
#         # print_gpu_utilization()
#         return super(CustomTrainer, self).training_step(*args, **kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print_gpu_utilization()

trainer.train()

print(trainer.evaluate())

# 218/654: {'eval_loss': 0.31987911462783813, 'eval_runtime': 148.4061, 'eval_samples_per_second': 2.54, 'eval_steps_per_second': 2.54, 'epoch': 1.0}
# 436/654: {'eval_loss': 0.2731330990791321, 'eval_runtime': 147.8519, 'eval_samples_per_second': 2.55, 'eval_steps_per_second': 2.55, 'epoch': 2.0}
# 654/654 {'eval_loss': 0.26411548256874084, 'eval_runtime': 147.7693, 'eval_samples_per_second': 2.551, 'eval_steps_per_second': 2.551, 'epoch': 3.0}
# {'train_runtime': 6601.1939, 'train_samples_per_second': 0.793, 'train_steps_per_second': 0.099, 'train_loss': 0.3134736518976521, 'epoch': 3.0}
