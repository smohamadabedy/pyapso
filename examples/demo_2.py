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


def constrained_benchmark(x):
    """
    3D constrained benchmark function with internal penalty handling.

    Objective:
        f(x) = x₀² + x₁² + x₂² + 10·sin(x₀)·cos(x₁)·sin(x₂)

    Constraint:
        x₀ + x₁ + x₂ ≤ 1.5

    Penalty:
        A large quadratic penalty is added if the constraint is violated.

    Parameters:
        x : array-like of shape (3,)

    Returns:
        float : Penalized objective function value.
    """
    x0, x1, x2 = x
    obj = x0**2 + x1**2 + x2**2 + 10 * np.sin(x0) * np.cos(x1) * np.sin(x2)

    # Constraint: x₀ + x₁ + x₂ ≤ 1.5
    g = x0 + x1 + x2 - 1.5
    penalty = 1e6 * max(0, g)**2

    return obj + penalty

    
# Run APSO and log results
if __name__ == "__main__":

    best, score, history, hd = APSO(
        objective_function=constrained_benchmark,
        dim=3,
        bounds=([-3, -2, -1], [3, 2, 1]),
        num_particles=100,
        max_iter=20,
        verbose = 1,folder="results_folder",save_prefix="results_file"
    ).run("avg")
    
    print("Best solution:", best)
    print("Best score:", score)
    
    # Save optimization history (JSON,CSV)
    hd.save()
    
    # Plot optimization history
    hd.plot()