from transformers import CodeLlamaTokenizerFast
from datasets import load_dataset
import torch

checkpoint = "./CodeLlama-13b-hf"

# Load, split, tokenize, collate dataset


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

print(dataset["train"])
print([(i, len(x["input_ids"])) for i, x in enumerate(dataset["train"])])
