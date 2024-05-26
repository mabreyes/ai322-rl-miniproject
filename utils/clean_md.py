from nbformat import read, write
import nbformat

# Path to your notebook
notebook_path = 'train_v2.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = read(f, as_version=nbformat.NO_CONVERT)

# Filter out markdown cells
notebook.cells = [cell for cell in notebook.cells if cell.cell_type != 'markdown']

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    write(notebook, f)
