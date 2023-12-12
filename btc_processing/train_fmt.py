import sys
import tempfile
import subprocess

def space(a, b):
    if a.isdigit() and b.isdigit(): return False
    if a in {"for", "if", "else", "range", "case", "return", "switch", "type", "break", "const", "chan", "defer"}: return True
    # if a.isalnum() and b.isalnum(): return True
    return False

cnt = 0
def gen_tgt(fname):
    global cnt
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
                    toks[i] = ""
                    i += 1
                    while not toks[i].startswith('"'):
                        toks[i] = ""
                        i += 1
                    continue
                elif toks[i] == "STR":
                    if toks[i + 1] == '"':
                        if i + 2 < len(toks) and toks[i + 2].startswith('"'):
                            toks[i + 1] = ""
                        elif i + 2 < len(toks) and (toks[i + 2] in {"NOT", "invalid", "no"}):
                            toks[i + 1] = ""
                            i += 2
                            while not toks[i].startswith('"'):
                                toks[i] = ""
                                i += 1
                            if toks[i] == '")':
                                toks[i] = '"'
                            continue
                    elif toks[i + 1] == '"\'"},':
                        toks[i + 1] = '"},'
                    elif toks[i + 1] == '""':
                        toks[i + 1] = '"'
                elif i + 1 < len(toks) and toks[i] in {"http", "https"} and toks[i + 1] == ":":
                    toks[i] = "STR"
                    toks[i + 1] = '"'
                elif i + 1 < len(toks) and toks[i].endswith("'") and toks[i + 1].startswith("'"):
                    toks[i] += " "
                elif i + 1 < len(toks) and space(toks[i], toks[i + 1]):
                    toks[i] += " "
                elif toks[i] == "int" and toks[i - 2] == "cells" and toks[i - 1] == "2":
                    pass
                elif toks[i] in {"struct", "string", "bool", "rune", "interface", "int", "float", "uint", "byte", "bytes", "Pool", "http", "strings", "sync", "Newtype", "BitArray", "IntSet", "Person", "time", "Node", "intHeap", "issue1304", "Drink", "error", "complex", "Industrious", "Sloth", "gResponse", "Contact", "Quadratic", "StopWatch", "Shape", "Dice", "Calculator", "wordQueue", "NotFoundError", "Stack", "ThreeStacks", "os", "defaultMatcher", "big", "Employee", "NumArray", "MyStack", "Wheel", "Solution", "Set", "atomic", "BytePool", "Stats", "Problem", "fmt", "ConnectFourBoard", "seqStack", "decoder", "StackHeap", "syscall", "BitMask", "Type", "ByteCounter", "uintptr", "IssuesSearchResult"}:
                    toks[i] = f" {toks[i]}"
                elif toks[i] == "Value" and toks[i - 1] != ".":
                    toks[i] = f" {toks[i]}"
                elif toks[i] == "var" and not toks[i + 1].isdigit():
                    toks[i] += " "
                elif toks[i] == "ListNode" and toks[i - 1] != "2":
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
                if len(p.stderr) != 0:
                    # cnt += 1
                    # if cnt > 5: continue
                    for j, l in enumerate(s.decode().split("\n")):
                        print(j, l)
                    print(p.stdout.decode())
                    print(p.stderr.decode())
                    print("\n" * 5)
                    raise Exception("parsing failed")
                lines = []
                for line in p.stdout.decode().split("\n"):
                    if line.startswith("\t"):
                        lines.append(line[1:])
                yield "\n".join(lines)

if __name__ == "__main__":
    for f in gen_tgt("go.train.tgt"):
        pass
    print(cnt)
        # print(f)
