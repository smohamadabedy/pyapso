from pyapso import APSO
import numpy as np

#======================================================================#
#======================================================================#
# Benchmark Objective Functions (with Constraints)                     #
#======================================================================#
# These functions are ready to be used with optimization algorithms.   #
#                                                                      #
#  Constraints are handled using penalty terms inside the objective    #
#  To perform multi-objective optimization, you can extend these       #
#     you can use the **weighted sum method**                          #
#     by combining objectives into a single scalar value:              #
#       f_total = w1 * obj1 + w2 * obj2 + ...                          #
#       where w1, w2, etc. are user-defined weights.                   #
#                                                                      #
#======================================================================#
#======================================================================#


def mccormick(x):
    """
    2D McCormick function with constraint-based bounding.

    Objective:
        f(x₁, x₂) = sin(x₁ + x₂) + (x₁ - x₂)² - 1.5·x₁ + 2.5·x₂ + 1

    Bounds (via penalty instead of clamping):
        -1.5 ≤ x₁ ≤ 5
        -3   ≤ x₂ ≤ 4

    Parameters:
        x : array-like of shape (2,)

    Returns:
        float : Penalized objective value.
    """
    x1, x2 = x

    # Apply boundary constraints using penalty
    if x1 < -1.5 or x1 > 5:
        return 1e6
    if x2 < -3 or x2 > 4:
        return 1e6

    obj = np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1
    return obj

    
# Run APSO and log results
if __name__ == "__main__":

    best, score, history, hd = APSO(
        objective_function=mccormick,
        dim=2,
        bounds=([-100,-100], [100,100]),
        num_particles=250,
        max_iter=30,
        verbose = 1,folder="results_folder",save_prefix="results_file"
    ).run("avg",c1=1.8, c2=1.8, w_min=0.5, w_max=0.9)
        
    print("Best solution:", best)
    print("Best score:", score)
    
    # Save optimization history (JSON,CSV)
    hd.save()
    
    # Plot optimization history
    hd.plot()