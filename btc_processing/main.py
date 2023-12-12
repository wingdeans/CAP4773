"""
from beautify import gen_src
from fmt import gen_tgt

import json

with open("test.json", "w") as f:
    for src, tgt in zip(gen_src("go.test.src"), gen_tgt("go.test.tgt")):
        json.dump({"src": src, "tgt": tgt}, f)
        f.write("\n")
"""
from beautify import gen_src
from train_fmt import gen_tgt

import json

dataset = "train"

with open(f"{dataset}.json", "w") as f:
    for src, tgt in zip(gen_src(f"go.{dataset}.src"), gen_tgt(f"go.{dataset}.tgt")):
        json.dump({"src": src, "tgt": tgt}, f)
        f.write("\n")
