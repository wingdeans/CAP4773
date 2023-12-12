"""
while read p; do
	gofmt -e <(echo -e "package main\nfunc main() {"; echo "$p" | sed -e 's/#/\n/g' -e 's/'; echo "}") 2>&1
	echo "$p" | sed 's/#/\n/g'
done < go.test.tgt
"""

import sys
import tempfile
import subprocess

def space(a, b):
    if a.isdigit() and b.isdigit(): return False
    if a in {"for", "if", "else", "range", "case", "var", "return", "switch", "type", "break", "const", "chan"}: return True
    # if a.isalnum() and b.isalnum(): return True
    return False

def gen_tgt(fname):
    with open(fname) as f:
        for line in f:
            toks = line.split()
            i = 0
            while i < len(toks):
                if toks[i] == "#":
                    toks[i] = "\n"
                # if toks[i].isdigit():
                #     j = i + 1
                #     while j < len(toks) and toks[j].isdigit():
                #         toks[i] += toks[j]
                #         toks[j] = ""
                #         j += 1
                #     i = j
                #     continue
                # if toks[i] == "'":
                #     j = i + 1
                #     while j < len(toks) and toks[j] != "'":
                elif toks[i] == '"%':
                    while not toks[i].startswith('"'):
                        toks[i] = ""
                        i += 1
                    continue
                elif toks[i] == "STR":
                    if toks[i + 1] == '"':
                        if i + 2 < len(toks) and toks[i + 2] == '"':
                            toks[i + 2] = ""
                        elif i + 2 < len(toks) and toks[i + 2] == "NOT":
                            i += 2
                            while not toks[i].startswith('"'):
                                toks[i] = ""
                                i += 1
                            continue
                elif i + 1 < len(toks) and toks[i].endswith("'") and toks[i + 1].startswith("'"):
                    toks[i] += " "
                elif i + 1 < len(toks) and space(toks[i], toks[i + 1]):
                    toks[i] += " "
                elif toks[i] in {"struct", "string", "bool", "rune", "interface", "int", "float", "uint", "byte", "bytes", "ListNode", "Pool", "http", "strings", "sync", "Newtype", "BitArray", "IntSet", "Person"}:
                    toks[i] = f" {toks[i]}"
                elif toks[i] in {"func", "<"}:
                    toks[i] = f" {toks[i]} "
    
                i += 1
    
            s = b"""
            package main
            func main() {
            """
            s += "".join(toks).encode()
            s += b"}"
    
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(s)
                temp.flush()
    
                p = subprocess.run(["gofmt", "-e", temp.name], capture_output=True)
                # if len(p.stderr) != 0:
                #     for j, l in enumerate(s.decode().split("\n")):
                #         print(j, l)
                #     print(p.stdout.decode())
                #     print(p.stderr.decode())
                #     print("\n" * 5)
                lines = []
                for line in p.stdout.decode().split("\n"):
                    if line.startswith("\t"):
                        lines.append(line[1:])
                yield "\n".join(lines)
