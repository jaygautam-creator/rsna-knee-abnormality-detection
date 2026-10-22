"""Convert a `# %%` cell-marked .py into a Kaggle-ready .ipynb.

Keeps notebooks reviewable as plain Python in git while still pushing real
notebooks to Kaggle. Usage: python -m knee.tools.py2nb <source.py> <out.ipynb>
"""

import json
import re
import sys

MARK = re.compile(r"^#\s*%%(.*)$")


def cells(text: str):
    cur, kind = [], "code"
    for line in text.splitlines():
        m = MARK.match(line)
        if m:
            if cur:
                yield kind, cur
            cur, kind = [], "markdown" if "markdown" in m.group(1) else "code"
        else:
            cur.append(line)
    if cur:
        yield kind, cur


def build(src: str) -> dict:
    out = []
    for kind, lines in cells(src):
        if kind == "markdown":
            body = [re.sub(r"^#\s?", "", ln) for ln in lines]
        else:
            body = lines
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            continue
        source = [ln + "\n" for ln in body[:-1]] + [body[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        nb = build(f.read())
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"{dst}: {len(nb['cells'])} cells")
