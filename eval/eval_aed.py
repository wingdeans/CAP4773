from tokenizers import Tokenizer

import re
re_whitespace = re.compile(r"\s+")

import editdistance
import json
from glob import glob

# root = "eval/old/checkpoint"
root = "eval/checkpoint"
checkpoints = sorted(glob(f"{root}*.json"), key=lambda s: int(s[len(root):s.rfind(".")]))
for checkpoint in checkpoints:
    total_ed, num_ed = 0, 0
    with open(checkpoint) as f:
        for line in f:
            line = json.loads(line[:-1])

            gen = re.sub(re_whitespace, " ", line["gen"].replace("\n", " # "))
            tgt = re.sub(re_whitespace, " ", line["tgt"].replace("\n", " # "))

            tokenizer = Tokenizer.from_file(f"btc/go/tokenized/tgttknizer.json")

            gen = tokenizer.encode(gen).tokens
            tgt = tokenizer.encode(tgt).tokens

            edis = min(editdistance.eval(tgt, gen), len(tgt))
            # edis = editdistance.eval(tgt, gen)
            # print(f"NED: {edis/len(tgt):.2f}")

            total_ed += edis/len(tgt)
            num_ed += 1
    print(f"AED SCORE: {total_ed / num_ed:.4f}")
