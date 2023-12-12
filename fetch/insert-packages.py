import sqlite3

con = sqlite3.connect("packages.db")
cur = con.cursor()

cur.execute("CREATE TABLE packages (name TEXT, src TEXT, deps TEXT, out TEXT)")
cur.execute("CREATE UNIQUE INDEX package_names ON packages (name)")

with open("packages") as f:
    packages = [(pkg[1:-1],) for pkg in f.read().split(" ")[1:-1] if pkg != '"go"' and not pkg.startswith('"go_')]

cur.executemany("INSERT INTO packages VALUES (?, NULL, NULL, NULL)", packages)
con.commit()
