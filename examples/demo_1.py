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


def rastrigin_5d(x):
    """
    5D Rastrigin function for high-dimensional optimization problems.

    Objective:
        f(x) = An + Σ [xᵢ² - A·cos(2πxᵢ)] for i = 1 to 5
        where A = 10

    Characteristics:
        - Highly multimodal (many local minima)
        - Global minimum at x = [0, 0, 0, 0, 0]

    Parameters:
        x : array-like of shape (5,)

    Returns:
        float : Objective value.
    """
    A = 10
    n = 5

    # Sum Rastrigin components across all dimensions
    sum_term = sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    obj = A * n + sum_term
    return obj
    
    
# Run APSO and log results
if __name__ == "__main__":

    # run for 5dim optimization problme with boundries
    best, score, history, hd = APSO(
        objective_function=rastrigin_5d,
        dim=5,
        bounds=([-5.12,-5.12,-5.12,-5.12,-5.12], [5.12,5.12,5.12,5.12,5.12]),
        num_particles=500,
        max_iter=40,
        verbose = 1,folder="results_folder",save_prefix="results_file"
    ).run(c1=1.8, c2=1.8,inertia=.05)    
        
    print("Best solution:", best)
    print("Best score:", score)
    
    # Save optimization history (JSON,CSV)
    hd.save()
    
    # Plot optimization history
    hd.plot()