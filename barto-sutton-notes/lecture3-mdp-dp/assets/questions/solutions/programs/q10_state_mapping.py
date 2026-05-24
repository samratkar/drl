"""
State Mapping: Lambda to convert (r, c) to 1D index i.
"""
import numpy as np

get_1d_index = lambda r, c, num_cols: r * num_cols + c

if __name__ == "__main__":
    r, c = 2, 3
    num_cols = 5
    index = get_1d_index(r, c, num_cols)
    print(f"Grid position ({r}, {c}) in a {num_cols}-column grid has 1D index: {index}")
