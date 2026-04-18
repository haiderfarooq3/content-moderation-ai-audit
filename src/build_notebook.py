"""Build a .ipynb file from a list of (cell_type, source) tuples."""
import json
import sys
from pathlib import Path


def make_notebook(cells, kernel_name="python3", display_name="Python 3"):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": display_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell_type, source in cells:
        src_lines = source.splitlines(keepends=True)
        if cell_type == "markdown":
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": src_lines,
                }
            )
        elif cell_type == "code":
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src_lines,
                }
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")
    return nb


def write_notebook(path, cells):
    nb = make_notebook(cells)
    Path(path).write_text(json.dumps(nb, indent=1))
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    print("usage: import make_notebook / write_notebook")
