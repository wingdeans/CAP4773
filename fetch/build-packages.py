import sqlite3
import subprocess

con = sqlite3.connect("packages.db")
cur = con.cursor()

packages = list(cur.execute("SELECT name FROM packages WHERE src IS NULL ORDER BY name"))
# packages = list(cur.execute('SELECT name FROM packages WHERE name > "woodpecker"'))

for i, (package,) in enumerate(packages):
    src = subprocess.run(
        ["nix", "build", "--print-out-paths", f"nixpkgs#{package}.src"],
        capture_output=True
    ).stdout.decode().strip()
    deps = subprocess.run(
        ["nix", "build", "--print-out-paths", f"nixpkgs#{package}.passthru.goModules"],
        capture_output=True
    ).stdout.decode().strip()
    out = subprocess.run(
        ["nix", "build", "--print-out-paths", f".#{package}"], capture_output=True
    ).stdout.decode().strip()
    cur.execute("UPDATE packages SET (src, deps, out) = (?, ?, ?) WHERE name = ?", (src, deps, out, package))
    con.commit()
    print(f"{i}/{len(packages)}", src, deps, out)
