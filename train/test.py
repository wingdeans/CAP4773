from transformers import CodeLlamaTokenizerFast, LlamaForCausalLM
import torch

checkpoint = "./CodeLlama-34b-hf"
device = "cuda"

tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint)
model = LlamaForCausalLM.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.bfloat16)

inputs = tokenizer.encode("""

""", return_tensors="pt").to(device)
outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500)

print(tokenizer.decode(outputs[0]))
