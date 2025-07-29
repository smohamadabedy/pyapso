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


def sphere(x):
    return np.sum(x**2)

    
# Run APSO and log results
def test_run_one_iteration_inertia():

    optimizer  = APSO(objective_function=sphere,dim=2,bounds=([-5,-5], [5,5]),num_particles=100,max_iter=1,verbose = 1,folder="results_folder",save_prefix="results_file")
    
    best_pos, best_score, history, _ = optimizer.run(mode="inertia")
    
    # Check types
    assert isinstance(best_pos, (list, np.ndarray)), "Best position should be list or ndarray"
    assert isinstance(best_score, float), "Best score should be float"
    
    # Check history has 1 record (since max_iter=1)
    assert len(history) == 1, "History should have one entry"

    # Check the best score is a finite number
    assert np.isfinite(best_score), "Best score should be a finite number"
    