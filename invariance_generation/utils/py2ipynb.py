# Source: https://gist.github.com/emielver/12419c73f02e12f138c344d5dd07ea45
import nbformat
import os
import sys

file_name = sys.argv[1]


def parse(code: list) -> nbformat.v4:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = []
    cell_value = ""

    for line in code:
        if line.startswith(("#%%", "# %%")):
            if cell_value:
                nb["cells"].append(nbformat.v4.new_code_cell(cell_value))
            cell_value = ""
        cell_value += line

    if cell_value:
        nb["cells"].append(nbformat.v4.new_code_cell(cell_value))
    return nb


with open(file_name) as file:
    nb = parse(file.readlines())
    ipynb = file_name.split(".py")[0] + ".ipynb"
    with open(ipynb, "w") as f:
        nbformat.write(nb, f)
        print("Generated: ", ipynb)
