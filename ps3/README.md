# CS229 Problem Set Instructions


## Setup for Written Parts

1. We have provided a LaTeX template in the `tex/` directory to make it easy to typeset your homework solutions.
2. Every problem has its own directory (*e.g.,* `tex/featuremaps/` for Problem 1).
3. Every subproblem has two files within the parent problem’s directory:
  - The problem statement, *e.g.* `tex/featuremaps/01-degree-3-math.tex` for Problem 1(a)). You do not need to modify this.
  - Your solution, *e.g.* `tex/featuremaps/01-degree-3-math-sol.tex` for your solution to Problem 1(a). You will need to modify these files (and the source files in `src` for coding parts).
4. You can use the given `Makefile` to typeset your solution, or use an editor with built-in typesetting such as TeXShop (comes free with the standard [LaTeX distribution](https://www.latex-project.org/get/)) or [Texpad](https://www.texpad.com/) (separate download, not free).


## Setup for Coding Parts

1. Set up a virtual environment using the environment manager of your choice. The setup used here is [uv](https://docs.astral.sh/uv/), a fast, Rust-based Python package and environment manager:
  - Run `uv venv` to create a `.venv` directory.
  - Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.
  - Run `uv pip install -r environment.txt` to install the required packages.
  - The Python interpreter is `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
  - Other tools such as Conda, `venv`, Poetry, or virtualenv are also fine; create an environment and install the packages listed in `environment.txt` with the tool you choose.
2. In VS Code or another editor, select the Python interpreter from the `.venv` directory created above. Do this each time you want to write/test your code.
3. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select the Python interpreter from the `.venv` directory you created
  - Select `OK` then `Apply`
4. Notice some coding problems come with `util.py` file. In it you have access to methods that do the following tasks:
  - Load a dataset in the CSV format provided in the problem
  - Add an intercept to a dataset (*i.e.,* add a new column of 1s to the design matrix)
  - Plot a dataset and a linear decision boundary. Some plots might require modified plotting code, but you can use this as a starting point.
5. Notice that start codes are provided in each problem directory (e.g. `gda.py`, `posonly.py`)
  - Within each starter file, there are highlighted regions of the code with the comments ** START CODE HERE ** and ** END CODE HERE **. You are strongly suggested to make your changes only within this region. You can add helper functions within this region as well.
