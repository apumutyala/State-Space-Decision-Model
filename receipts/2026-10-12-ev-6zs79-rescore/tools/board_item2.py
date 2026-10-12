"""Replace BOARD.md item 2's status cell. Usage: board_item2.py <BOARD.md> <new status text>"""
import sys

p, status = sys.argv[1], sys.argv[2]
b = open(p, encoding="utf-8").read()
lines = b.split("\n")
hit = 0
for i, ln in enumerate(lines):
    if ln.startswith("| 2 | Kit parity"):
        cells = ln.split("|")
        cells[-2] = f" {status} "
        lines[i] = "|".join(cells)
        hit += 1
assert hit == 1, hit
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print([ln for ln in lines if ln.startswith("| 2 | Kit parity")][0])
