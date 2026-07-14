"""
Visualizing the Value Function: Reshape and print 1D array as 4x4 matrix.
"""
import numpy as np

def visualize_v_4x4(V_array):
    """
    Assumes V_array is a flat numpy array or list of 16 elements.
    """
    V_matrix = np.array(V_array).reshape(4, 4)
    print("Value Function Visualization (4x4):")
    print(np.round(V_matrix, 2))

if __name__ == "__main__":
    # Example visualization call
    V_dummy = np.linspace(0, 1.5, 16)
    visualize_v_4x4(V_dummy)
