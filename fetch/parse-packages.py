import os
import json
import sqlite3
from elftools.elf.elffile import ELFFile
from collections import defaultdict
from operator import itemgetter
from glob import iglob
from pathlib import Path
from tree_sitter import Language, Parser

Language.build_library("tree-sitter-go.so", ["tree-sitter-go"])
GoLanguage = Language("tree-sitter-go.so", "go")

parser = Parser()
parser.set_language(GoLanguage)

query = GoLanguage.query("[(interpreted_string_literal) @s (raw_string_literal) @s (comment) @c]")

con = sqlite3.connect("packages.db")
topcur = con.cursor()
cur = con.cursor()

topcur.execute("CREATE TABLE IF NOT EXISTS functions (package TEXT, "
    "src_path TEXT, linum INTEGER, name TEXT, src TEXT, "
    "bin_path TEXT, pc_lo INTEGER, pc_hi INTEGER, decomp TEXT, "
    "CONSTRAINT sources_unique UNIQUE (bin_path, pc_lo))")
con.commit()

abnormalities = defaultdict(set)

def parse(f, binpath, pkgname, src, deps, out):
    elf = ELFFile(f)
    dwarf = elf.get_dwarf_info()
    for CU in dwarf.iter_CUs():
        lp = dwarf.line_program_for_CU(CU)
        for DIE in CU.iter_DIEs():
            if DIE.tag != "DW_TAG_subprogram": continue
            try:
                name, file_idx, linum, pc_lo, pc_hi = itemgetter(
                    "DW_AT_name", "DW_AT_decl_file", "DW_AT_decl_line",
                    "DW_AT_low_pc", "DW_AT_high_pc"
                )(DIE.attributes)
                name, file_idx, linum = name.value.decode(), file_idx.value, linum.value
                pc_lo, pc_hi = pc_lo.value, pc_hi.value
            except: continue
            if ".." in name or file_idx == 0 or linum == 0: continue

            file_entry = lp["file_entry"][file_idx]
            filename = file_entry.name.decode().strip()
            if not filename.endswith(".go"): continue

            dirname = lp["include_directory"][file_entry.dir_index].decode().strip()
            if dirname.startswith("./vendor/"):
                if deps:
                    path = Path(deps, dirname[len("./vendor/"):], filename)
                else:
                    path = Path(src, dirname, filename)
            elif dirname == "." or dirname.startswith("./"):
                path = Path(src, dirname, filename)
            elif dirname.startswith("/"):
                abnormalities["absolute_paths"].add(dirname)
                continue
            else:
                raise Exception(f"Invalid dirname: {dirname}")

            try:
                with open(path, "rb") as f:
                    text = f.read()
                    ast = parser.parse(text)

                    node = ast.root_node.named_descendant_for_point_range((linum - 1, 0), (linum - 1, 1))
                    while node and node.type not in ["function_declaration", "method_declaration"]:
                        node = node.parent

                    if node:
                        # print(name)

                        ident_node = node.child_by_field_name("name")
                        ident = text[ident_node.start_byte:ident_node.end_byte].decode()
                        basename = name[name.rfind(".") + 1:]
                        if ident != basename:
                            abnormalities["mismatches"].add(f"{ident} ({path}:{linum}) != {basename}")
                            continue

                        func, idx = bytearray(), node.start_byte
                        for c, t in query.captures(ast.root_node, start_byte=node.start_byte, end_byte=node.end_byte):
                            func += text[idx:c.start_byte]
                            if t == "s":
                                func += b'"STR"'
                            idx = c.end_byte
                        func += text[idx:node.end_byte]

                        # "(package TEXT, "
                        # "src_path TEXT, linum INTEGER, name TEXT, src TEXT, "
                        # "bin_path TEXT, pc_lo INTEGER, pc_hi INTEGER, decomp TEXT)"

                        cur.execute("INSERT OR IGNORE INTO functions VALUES (?, "
                            "?, ?, ?, ?, "
                            "?, ?, ?, NULL)",
                            (pkgname,
                            str(path), linum, name, func,
                            binpath, pc_lo, pc_hi)
                        )
                        # print(func.decode())
                        # print(text[node.start_byte:node.end_byte].decode())
                    else:
                        abnormalities["nodeless"].add(name)
            except Exception as e:
                print(e)
                abnormalities["exceptions"].add(str(e))

with open("modroots.json") as f:
    modroots = json.load(f)
modroots["documize-community"] = "edition"
modroots["ncdns"] = "ncdns"

skip = {
    "filegive", "honk", "reposurgeon", # compressed sources
    "mop", # swapped directories
    "opensnitch", # grpc
    "writefreely", # assets
    "xe-guest-utilities", "yubihsm-connector" # prebuild
}

for (name, src, deps, out) in topcur.execute("SELECT * FROM packages WHERE out IS NOT NULL AND out <> '' ORDER BY name"):
# for (name, src, deps, out) in topcur.execute("SELECT * FROM packages WHERE out IS NOT NULL AND out <> '' AND name >= 'ytcast' ORDER BY name"):
    if name in skip: continue

    print(name, src, deps, out)
    if name in modroots:
        src = os.path.join(src, modroots[name])
    for fname in iglob(f"{out}/**", recursive=True, include_hidden=True):
        if not os.path.isfile(fname): continue
        with open(fname, "rb") as f:
            if f.read(4) != b"\x7fELF": continue
            parse(f, fname, name, src, deps, out)
    con.commit()
    #print(src, deps, out)

print(abnormalities)
