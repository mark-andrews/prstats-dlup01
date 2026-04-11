# Software Requirements and Installation

Setting up Python for deep learning is more involved than installing a single application.
There are many valid combinations of tools, and this document describes the most common approaches.
All software listed here is free and open-source.
Python 3.9 or later is required.

## Choosing an Interactive Environment

For the hands-on coding in this course you need an environment that lets you run Python interactively.
There are two main categories worth considering: a data science IDE, or Jupyter.

### Option 1: An IDE

An integrated development environment gives you an editor, an interactive console, a variable explorer, and a plot viewer in one application.
My recommendation is [Positron](https://positron.posit.co/), an open-source data science IDE developed by Posit, the company behind RStudio.
It is designed specifically for interactive data work in Python and R, and will feel immediately familiar to anyone coming from RStudio.

Other IDEs are equally valid if you already have a preference:

- [Spyder](https://www.spyder-ide.org/) is another data science-oriented option.
- [VS Code](https://code.visualstudio.com/) with the Python extension is widely used.
- [PyCharm](https://www.jetbrains.com/pycharm/) is a full-featured option popular in professional settings.

### Option 2: Jupyter

Jupyter is a browser-based environment in which you write and run Python inside a notebook: a document that interleaves code cells, their output, and prose.
It is widely used in data science and scientific computing and works well for the exploratory and experimental work this course involves.

There are two main ways to use Jupyter locally:

- **Jupyter Notebook** is the original interface, simple and lightweight.
- **JupyterLab** is its modern successor, offering a more complete environment with a file browser, multiple panels, and a terminal.

For a local installation, JupyterLab is the better choice; instructions are at [jupyter.org/install](https://jupyter.org/install).

**Google Colab** is a cloud-hosted Jupyter environment at [colab.research.google.com](https://colab.research.google.com).
It runs entirely in the browser, requires no local installation, and provides free GPU access, which is particularly useful for deep learning.
It is the simplest option if you want to avoid local setup or if your machine does not have a GPU.

## A Note on GPUs

Deep learning training is significantly faster on a GPU.
All examples in this course can run on CPU, so a GPU is not required.
Where training times would be prohibitive on CPU, we use small datasets and small models that remain practical.
If you have an NVIDIA GPU, PyTorch will use it automatically once CUDA is installed (see the PyTorch installation instructions below).
If you do not have a local GPU, Google Colab is the recommended alternative.

## Required Packages

The course requires the following Python packages:

```
torch  torchvision  numpy  matplotlib  scikit-learn  skorch  transformers  tokenizers  datasets  jupyter  ipykernel
```

### Installing PyTorch

PyTorch installation depends on your hardware and operating system.
The recommended way to get the right version is to use the official installation selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
For most participants on CPU-only machines, the command will be:

```bash
pip install torch torchvision
```

For NVIDIA GPU support, the site will generate the appropriate command including the CUDA version matching your drivers.

### Installing the remaining packages

Once PyTorch is installed, install the remaining packages:

```bash
pip install numpy matplotlib scikit-learn skorch transformers tokenizers datasets jupyter ipykernel
```

It is strongly recommended to install all packages into a virtual environment (see below).
If you are using Google Colab, PyTorch, NumPy, and most other packages are already available; install any missing ones with `pip install` in a notebook cell.

## Virtual Environments

A virtual environment is an isolated Python installation scoped to a specific project.
It prevents packages installed for one project from conflicting with those of another, and keeps your system Python clean.
This is standard practice in Python and worth using from the start.

### Using venv (recommended)

Python's built-in `venv` module requires no additional installation:

```bash
# Create a virtual environment in a folder named 'env'
python -m venv env

# Activate it — macOS and Linux
source env/bin/activate

# Activate it — Windows
env\Scripts\activate

# Install PyTorch (CPU version — visit pytorch.org for GPU version)
pip install torch torchvision

# Install the remaining packages
pip install numpy matplotlib scikit-learn skorch transformers tokenizers datasets jupyter ipykernel

# Deactivate when finished
deactivate
```

Once the environment is activated, your terminal prompt will show its name, and all `pip install` commands will install into it rather than system-wide.
In Positron or VS Code, you select the virtual environment as your Python interpreter and the IDE handles activation for you.

### Using conda

Conda is an alternative package and environment manager popular in the scientific Python community.
If you already have Anaconda or Miniconda installed, you can create an environment as follows:

```bash
conda create -n pydl python=3.11
conda activate pydl
pip install torch torchvision
pip install numpy matplotlib scikit-learn skorch transformers tokenizers datasets jupyter ipykernel
```

Note that PyTorch is installed via pip even within a conda environment, to ensure you get the correct version from the PyTorch installer.

## Suggested Setup

The simplest local setup that covers everything needed for this course:

1. Install Python 3.9 or later from [python.org/downloads](https://www.python.org/downloads/)
2. Create a virtual environment: `python -m venv env` and activate it
3. Install PyTorch using the command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
4. Install remaining packages: `pip install numpy matplotlib scikit-learn skorch transformers tokenizers datasets jupyter ipykernel`
5. Install [Positron](https://positron.posit.co/) and point it at the virtual environment as your interpreter

Alternatively, use Google Colab with no local installation at all.
Colab provides GPU access, all common packages pre-installed, and a familiar Jupyter interface.

There is no single correct way to do this.
The combination of Python, a virtual environment, the required packages, and a comfortable editor covers all possible approaches.
