from transformers import CodeLlamaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import json
from glob import glob

checkpoint = "./CodeLlama-7b-hf"
# adapter = "./train/long/checkpoint-9001"

# Load, split, tokenize, collate dataset

dataset = load_dataset("json", data_files={"test": "train/test.json"})
# dataset = dataset["train"].train_test_split(test_size=0.2)

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(pairs):
    return tokenizer([f"<ASM>\n{src}</ASM>\nfunc main() {{\n" for src in pairs["src"]])

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=4,
    remove_columns=["src"]# dataset["train"].column_names
)

dataset = dataset.filter(lambda x: len(x["input_ids"]) < 7000)

# Load model

from transformers import LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel

pre_model = LlamaForCausalLM.from_pretrained(
    checkpoint, device_map={"": 0}, load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

adapters = sorted(glob("train/long/*"), key=lambda s: int(s[len("train/long/checkpoint-"):]))
for i, adapter in enumerate(adapters):
    model = PeftModel.from_pretrained(pre_model, adapter)

    with torch.no_grad():
        for test in dataset["test"]:
            prompt = torch.tensor([test["input_ids"]])
            # print(tokenizer.batch_decode(prompt))
            out = model.generate(
                input_ids=prompt.to("cuda"),
                attention_mask=torch.tensor([test["attention_mask"]], dtype=torch.bool),
                max_new_tokens=500,
                eos_token_id=9891,
                pad_token_id=2
            )
            out = tokenizer.batch_decode(out[:, prompt.shape[1]:])[0] # skip_special_tokens=True))
            if len(out) > 0:
                out = out.split("\n")
                if out[-1].strip() == "func":
                    del out[-1]
                out = "\n".join(line[1:] if line and line[0] == "\t" else line for line in out)
    
            with open(f"eval/checkpoint{i}.json", "a") as f:
                json.dump({"tgt": test["tgt"], "gen": out}, f)
                f.write("\n")
