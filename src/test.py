from src_pyapso import APSO
import numpy as np

def fitness(x):
    return np.sum(x**2)

if __name__ == "__main__":
    best, score, history, hd = APSO(
        objective_function=fitness,
        dim=1,
        bounds=([-10], [10]),
        num_particles=100,
        max_iter=10,
        verbose = 1,live_plot=False,folder="results_folder",save_prefix="results_file"
    ).run()    
            
    print("Best solution:", best)
    print("Best score:", score)

    # Save optimization history (JSON,CSV)
    hd.save()

    # Plot optimization history
    hd.plot()