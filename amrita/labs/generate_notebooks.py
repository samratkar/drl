import json
import os
import sys
import io
import contextlib
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_code_cell(code_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_str.strip().splitlines(keepends=True)
    }

def create_markdown_cell(md_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": md_str.strip().splitlines(keepends=True)
    }

def execute_and_populate_notebook(cells, global_env=None):
    if global_env is None:
        global_env = {}
    
    # Custom display and print capture
    execution_count = 1
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = execution_count
            code_text = "".join(cell["source"])
            
            stdout_io = io.StringIO()
            stderr_io = io.StringIO()
            plt.close('all')
            
            outputs = []
            try:
                with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                    # We execute the cell
                    exec(code_text, global_env)
                
                stdout_text = stdout_io.getvalue()
                stderr_text = stderr_io.getvalue()
                
                if stdout_text:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": stdout_text.splitlines(keepends=True)
                    })
                if stderr_text:
                    outputs.append({
                        "name": "stderr",
                        "output_type": "stream",
                        "text": stderr_text.splitlines(keepends=True)
                    })
                
                # Check if a matplotlib figure was created
                fig_nums = plt.get_fignums()
                for fignum in fig_nums:
                    fig = plt.figure(fignum)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
                    img_buf.seek(0)
                    img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
                    outputs.append({
                        "data": {
                            "image/png": img_b64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close(fig)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                outputs.append({
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": tb.splitlines(keepends=True)
                })
                print(f"Error in cell execution:\n{code_text}\nError: {e}")
            
            cell["outputs"] = outputs
            execution_count += 1
            
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb

def save_notebook(nb, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved: {filepath}")

print("Notebook execution engine ready.")
