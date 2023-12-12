from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

checkpoint = "./CodeLlama-13b-hf"

# Load, split, tokenize, collate dataset

print_gpu_utilization()

dataset = load_dataset("json", data_files="train/test.json")
dataset = dataset["train"].train_test_split(test_size=0.2)

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)

def preprocess(pairs):
    return tokenizer([f"<ASM>\n{src}</ASM>\n{tgt}"
                      for src, tgt in zip(pairs["src"], pairs["tgt"])])

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=10,
    remove_columns=dataset["train"].column_names
)

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
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    bf16=True,
)

torch.cuda.memory._record_memory_history()

i = 0
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        global i
        if i == 80:
            torch.cuda.memory._dump_snapshot("snapshot2.pickle")
        i += 1
        # print(args, kwargs)
        print_gpu_utilization()
        return super(CustomTrainer, self).training_step(*args, **kwargs)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print_gpu_utilization()

trainer.train()

print(trainer.evaluate())
