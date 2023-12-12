from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import textwrap

checkpoint = "./CodeLlama-7b-hf"

# Load, split, tokenize, collate dataset

dataset = load_dataset("json", data_files="train/test.json")
dataset = dataset["train"].train_test_split(test_size=0.2)

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

dataset = dataset.filter(lambda x: len(x["input_ids"]) < 7000)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

print(dataset)

# Load model

from transformers import LlamaForCausalLM, TrainingArguments, Trainer
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
    num_train_epochs=6,
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

trainer.train()

"""
Map (num_proc=4): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 584/584 [00:01<00:00, 558.60 examples/s]
Map (num_proc=4): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 147/147 [00:00<00:00, 417.86 examples/s]
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 584/584 [00:01<00:00, 480.57 examples/s]
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 147/147 [00:00<00:00, 451.03 examples/s]
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 557
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 142
    })
})
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:21<00:00, 10.53s/it]
trainable params: 4,194,304 || all params: 6,742,740,992 || trainable%: 0.06220473254091146
  0%|                                                                                                                                                                           | 0/207 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
/blue/cap4773/z.liu1/conda/lib/python3.11/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'eval_loss': 0.4430771470069885, 'eval_runtime': 105.6346, 'eval_samples_per_second': 1.344, 'eval_steps_per_second': 1.344, 'epoch': 0.99}
 33%|██████████████████████████████████████████████████████                                                                                                            | 69/207 [22:07<35:07, 15.27s/it/blue/cap4773/z.liu1/conda/lib/python3.11/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'eval_loss': 0.40218159556388855, 'eval_runtime': 105.5885, 'eval_samples_per_second': 1.345, 'eval_steps_per_second': 1.345, 'epoch': 2.0}
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                     | 139/207 [43:54<17:26, 15.39s/it/blue/cap4773/z.liu1/conda/lib/python3.11/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'eval_loss': 0.38823747634887695, 'eval_runtime': 105.7234, 'eval_samples_per_second': 1.343, 'eval_steps_per_second': 1.343, 'epoch': 2.97}
{'train_runtime': 3912.8821, 'train_samples_per_second': 0.427, 'train_steps_per_second': 0.053, 'train_loss': 0.45031093284127793, 'epoch': 2.97}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 207/207 [1:05:12<00:00, 18.90s/it]"""
