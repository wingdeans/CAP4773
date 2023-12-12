import sys
import io

output = None

def P(*args):
    print(*args, file=output, end="")

def parse_unsigned(L, i):
    while L[i].isdigit():
        P(L[i])
        i += 1
    return i

def parse_signed(L, i):
    if L[i] == "-":
        P("-")
        i += 1
    return parse_unsigned(L, i)

known_regs = [
    "fs", "rsp", "rbp", "rip",
    "rax", "rbx", "rcx", "rdx",
    "eax", "ebx", "ecx", "edx",
    "ax", "ah", "al", "bx", "bh", "bl", "ch", "cl", "cx", "dx", "dl",
    "rdi", "edi", "rsi", "esi",
    "si", "sil", "di", "dil"
]
def parse_reg(L, i):
    assert L[i] in ["(%", "%", ",%", "(,%", "*%"]
    P(L[i])
    i += 1
    if L[i] in ["r", "xmm"]:
        P(L[i])
        i = parse_unsigned(L, i + 1)
        if L[i] in ["d", "b"]:
            P(L[i])
            i += 1
        return i
    else:
        assert L[i] in known_regs, L[i]
        P(L[i])
        return i + 1

def parse_rm(L, i):
    if is_sym(L[i]):
        i = parse_sym(L, i)
    elif L[i] in ["%", "*%"]:
        i = parse_reg(L, i)

        if L[i] != ":":
            return i

        P(":")
        i += 1

    if i >= len(L):
        return i

    i = parse_signed(L, i)
    if L[i] in ["(%", "(,%"]:
        i = parse_reg(L, i)

        if L[i] == ",%":
            i = parse_reg(L, i)

        if L[i] == ",":
            P(L[i])
            i = parse_unsigned(L, i + 1)

    if L[i] == ")":
        P(")")
        i += 1

    return i

def parse_imm(L, i):
    assert L[i] in ["$", "$-"], L[i]
    P(L[i])
    return parse_unsigned(L, i + 1)

def is_sym(s):
    return all(c.isalnum() or c in "_.+@" for c in s)

# exceptions = [["eval", ".", "call"], ["main", ".", "ret"]]
syms = []
def parse_sym(L, i):
    start = i
    while i < len(L) and (is_sym(L[i]) and (L[i] not in instructions or L[i-1] == ".")):
        i += 1

    j = i
    while j < len(L) and L[j].isalnum():
        j += 1
    if j < len(L) and L[j] == ":":
        while L[i] != ".":
            i -= 1

    sym = "".join(L[start:i])
    syms.append(sym)
    P(f"<{sym}>")
    return i

def parse_imm_or_rm(L, i):
    if L[i] in ["$", "$-"]:
        return parse_imm(L, i)
    else:
        return parse_rm(L, i)

instructions = {
    "leaq", "cmpb", "cmpl", "cmpw", "cmpq", "testq", "testl", "testb",
    "call"
}
jmp_insns = {
    "jnb", "jmp", "jge", "jle", "js", "jbe", "jg", "jl", "jne", "je", "ja", "jns", "jp", "jb"
}
two_op_insns = {
    "movl", "movq", "movb", "movabsq", "movzbl", "movzwl", "movdqa", "movups", "movsd", "movw",
    "movdqu", "movaps", "movapd", "movslq", "movsq", "movswq", "movss", "movd", "movsbq",
    "addl", "addq", "addsd", "addb", "addss",
    "subl", "subq", "subsd", "subb", "subss",
    "andq", "andl", "andw", "andb",
    "orq", "orw", "orl", "orb",
    "pxor", "xorq", "xorb", "xorl", "xorw", "xorpd",
    "cmpb", "cmpl", "cmpw", "cmpq",
    "testq", "testl", "testb", "testw",
    "mulsd", "divsd", "divss", "leaq", "leal",
    "cmovg", "cmovle", "cmovs", "cmove", "cmovne",
    "ucomisd", "comisd", "ucomiss", "comiss",
    "xchgq", "xchgl", "xaddl", "bsrq"
}
var_op_insns = {
    "shrl", "shrq", "salq", "shrb", "sarq", "sall", "shrw", "sarl",
    "roll",
    "imulq", "imull", "mulq"
}
one_op_insns = {
    "pushq", "popq",
    "setne", "sete", "setge", "setbe", "seta", "setle", "setnp",
    "setg", "setl", "setnb", "setp", "setb",
    "negq", "negl", "notq", "notl",
    "idivq", "divq", "divb", "idivl", "divl"
}
zero_op_insns = {"ret", "leave", "nop", "cqto", "cltq", "cltd", "stosq", "lock"}
cvt_insns = {"cvtsi", "cvttsd", "cvttss"}

instructions |= jmp_insns
instructions |= two_op_insns
instructions |= one_op_insns
instructions |= zero_op_insns
instructions |= var_op_insns
instructions |= cvt_insns

unknown = set()
def gen_src(fname):
    global output

    with open(fname) as f:
        for linum, line in enumerate(f):
            output = io.StringIO()

            # print(linum)
            L = line.split()
            i = 0
            while i < len(L):
                if L[i] == ".":
                    while L[i] != ":":
                        P(L[i])
                        i += 1
                    P(":")
                    i += 1
                else:
                    if L[i] in zero_op_insns:
                        print(L[i], file=output, end="")
                        i += 1
                    else:
                        print(L[i], file=output, end=" ")
                        if L[i] in jmp_insns:
                            assert L[i + 1] == "*%" or L[i + 1:i + 3] == [".", "L"], L[i + 1:i + 3]
                            P(L[i + 1] + L[i + 2])
                            i = parse_unsigned(L, i + 3)
                        elif L[i] in two_op_insns:
                            i = parse_imm_or_rm(L, i + 1)
                            assert L[i] in [",", "),"], L[i]
                            P(L[i])
                            i = parse_rm(L, i + 1)
                        elif L[i] in one_op_insns:
                            i = parse_imm_or_rm(L, i + 1)
                        elif L[i] in zero_op_insns:
                            i += 1
                        elif L[i] in var_op_insns:
                            i = parse_imm_or_rm(L, i + 1)
                            if L[i] in [",", "),"]:
                                P(L[i])
                                i = parse_rm(L, i + 1)
                            if L[i] in [",", "),"]:
                                P(L[i])
                                i = parse_rm(L, i + 1)
                        elif L[i] in cvt_insns:
                            assert L[i + 1] == "2", L[i + 1]
                            assert L[i + 2] in ["sdq", "siq", "sdl", "ssq", "sil"], L[i + 2]
                            P("".join(L[i + 1:i + 3]))
                            i = parse_rm(L, i + 3)
                            assert L[i] in [",", "),"], L[i]
                            P(L[i])
                            i = parse_rm(L, i + 1)
                        elif L[i] == "call":
                            i = parse_rm(L, i + 1)
                        elif L[i] == "rep":
                            assert L[i + 1] in instructions, L[i + 1]
                            P(L[i + 1])
                            i += 2
                            continue
                        else:
                            raise Exception(L[i])
                            # unknown.add(L[i])
                            break
                print(file=output)

            yield output.getvalue().strip()

# print(unknown)
# 
# for sym in syms:
#     print(sym)
