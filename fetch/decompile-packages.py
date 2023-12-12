import os
import sqlite3
import subprocess
from glob import iglob

analyzeHeadless = "/nix/store/2b38sgpx9mzpxkxvp7flima884xjpi3b-ghidra-10.3.3/lib/ghidra/support/analyzeHeadless"

os.environ["LD_PRELOAD"] = "/lib64/libnss_sss.so.2"
def decompile(path):
    subprocess.run([analyzeHeadless, ".", "ghidra", "-import", path])

con = sqlite3.connect("packages.db")
cur = con.cursor()

for (out,) in cur.execute("SELECT out FROM packages WHERE out IS NOT NULL AND out <> '' ORDER BY name"):
    for fname in iglob(f"{out}/**", recursive=True, include_hidden=True):
        if not os.path.isfile(fname): continue
        with open(fname, "rb") as f:
            if f.read(4) != b"\x7fELF": continue
            decompile(fname)
