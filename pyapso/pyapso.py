# =============================================================
# =============================================================
#
#
#        ░█████╗░██████╗░░█████╗░██████╗░████████╗██╗██╗░░░██╗███████╗
#        ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██║░░░██║██╔════╝
#        ███████║██║░░██║███████║██████╔╝░░░██║░░░██║╚██╗░██╔╝█████╗░░
#        ██╔══██║██║░░██║██╔══██║██╔═══╝░░░░██║░░░██║░╚████╔╝░██╔══╝░░
#        ██║░░██║██████╔╝██║░░██║██║░░░░░░░░██║░░░██║░░╚██╔╝░░███████╗
#        ╚═╝░░╚═╝╚═════╝░╚═╝░░╚═╝╚═╝░░░░░░░░╚═╝░░░╚═╝░░░╚═╝░░░╚══════╝
#
#        ██████╗░░█████╗░██████╗░████████╗██╗░█████╗░██╗░░░░░███████╗  
#        ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║░░░░░██╔════╝  
#        ██████╔╝███████║██████╔╝░░░██║░░░██║██║░░╚═╝██║░░░░░█████╗░░  
#        ██╔═══╝░██╔══██║██╔══██╗░░░██║░░░██║██║░░██╗██║░░░░░██╔══╝░░  
#        ██║░░░░░██║░░██║██║░░██║░░░██║░░░██║╚█████╔╝███████╗███████╗  
#        ╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚══════╝╚══════╝  
#
#        ░██████╗░██╗░░░░░░░██╗░█████╗░██████╗░███╗░░░███╗
#        ██╔════╝░██║░░██╗░░██║██╔══██╗██╔══██╗████╗░████║
#        ╚█████╗░░╚██╗████╗██╔╝███████║██████╔╝██╔████╔██║
#        ░╚═══██╗░░████╔═████║░██╔══██║██╔══██╗██║╚██╔╝██║
#        ██████╔╝░░╚██╔╝░╚██╔╝░██║░░██║██║░░██║██║░╚═╝░██║
#        ╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝
#
#        ░█████╗░██████╗░████████╗██╗███╗░░░███╗██╗███████╗░█████╗░████████╗██╗░█████╗░███╗░░██╗
#        ██╔══██╗██╔══██╗╚══██╔══╝██║████╗░████║██║╚════██║██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
#        ██║░░██║██████╔╝░░░██║░░░██║██╔████╔██║██║░░███╔═╝███████║░░░██║░░░██║██║░░██║██╔██╗██║
#        ██║░░██║██╔═══╝░░░░██║░░░██║██║╚██╔╝██║██║██╔══╝░░██╔══██║░░░██║░░░██║██║░░██║██║╚████║
#        ╚█████╔╝██║░░░░░░░░██║░░░██║██║░╚═╝░██║██║███████╗██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
#        ░╚════╝░╚═╝░░░░░░░░╚═╝░░░╚═╝╚═╝░░░░░╚═╝╚═╝╚══════╝╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝
#
#
#  Adaptive Particle Swarm Optimization (APSO)
#  - Parallel and vectorized evaluation for faster optimization
#  - Logs results to CSV and JSON
#  - Visualizes optimization performance
# =============================================================
# =============================================================

import json
import csv
import sys
import json
import csv
import os

import numpy                as np
import multiprocessing      as mp
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec

from datetime   import datetime 
from tqdm       import tqdm      
from time       import time        

plt.rcParams.update({'font.size': 8})


# APSO implementation for general n-dimensional optimization
class APSO:
    def __init__(self, objective_function, dim=2,bounds=(-5.12, 5.12),num_particles=30,max_iter=100,verbose = 1,live_plot=False,folder="results_folder",save_prefix="results_file"):
 
        """
        Initialize the Adaptive Particle Swarm Optimization (APSO) solver.

        Parameters
        ----------
        objective_function : callable
            The objective function to be minimized. Must accept a 1D array of floats
            as input and return a scalar float (the cost or fitness value).

        dim : int, optional (default=2)
            Number of dimensions (design variables) in the search space.

        bounds : tuple of array-like or floats, optional (default=(-5.12, 5.12))
            Lower and upper bounds for the design variables. Can be either:
            - A tuple of two floats, applied uniformly across all dimensions.
            - A tuple of two lists of length `dim`, specifying element-wise bounds.

        num_particles : int, optional (default=30)
            Number of particles in the swarm. Larger swarms improve exploration
            but increase computation time.

        max_iter : int, optional (default=100)
            Maximum number of iterations (generations) to perform.

        verbose : int, optional (default=1)
            Verbosity level. Set to 1 to print progress; 0 to suppress output.

        live_plot : bool, optional (default=False)
            If True, shows a live plot of convergence (only available for 2D cases).

        folder : str, optional (default="results_folder")
            Directory where the results (history, best score, plots) will be saved.

        save_prefix : str, optional (default="results_file")
            Prefix for result files (used in saved CSV/JSON/plot files).

        Notes
        -----
        - For multi-objective problems, consider extending the objective function 
          using weighted-sum.
        - This class supports constrained optimization via penalty-based methods.
        """
    
        self.timestamp          = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.objective_function = objective_function
        self.dim                = dim
        self.bounds             = bounds
        self.num_particles      = num_particles
        self.max_iter           = max_iter
        self.verbose            = verbose
        self.history            = []
        self.folder             = folder
        self.save_prefix        = save_prefix
        self.live_plot          = live_plot

        os.makedirs(f"{self.folder}_{self.timestamp}", exist_ok=True)
        
        self.json_path  = os.path.join(f"{self.folder}_{self.timestamp}", f"{self.save_prefix}_{self.timestamp}.json")
        self.csv_path   = os.path.join(f"{self.folder}_{self.timestamp}", f"{self.save_prefix}_{self.timestamp}.csv")
        
        lb, ub = self.bounds
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
        
        if self.live_plot:
            self._init_plot()
            
    # Evaluate a batch of positions in parallel
    def evaluate_population(self,population, func):
        with mp.Pool() as pool:
            fitness = pool.map(func, population)
        return np.array(fitness)
    
    # Initialize a real-time optimization progress plot 
    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(6, 3))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2])

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax4 = self.fig.add_subplot(gs[1, :])

        self.line1, = self.ax1.plot([], [], color='blue')
        self.line2, = self.ax2.plot([], [], color='green', linestyle='--')
        self.line3, = self.ax3.plot([], [], color='orange')
        self.line4, = self.ax4.plot([], [], label="Best Position Mean")
        self.line5, = self.ax4.plot([], [], label="Best Position Std", linestyle='--')

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True)
        self.ax1.set_title("Best Score")
        self.ax2.set_title("Mean Score")
        self.ax3.set_title("Fitness Standard Deviation")
        self.ax4.set_title("Best Position Statistics")
        self.ax1.set_ylabel("Best Score")
        self.ax2.set_ylabel("Mean Score")
        self.ax3.set_ylabel("Std Score")
        self.ax4.set_ylabel("Position Stats")
        self.ax4.set_xlabel("Iteration")
        self.ax4.legend()
        self.fig.tight_layout()
     
    # Update the optimization progress plot 
    def _update_plot(self):
        iterations  = [row["iteration"] for row in self.history]
        best_scores = [row["best_score"] for row in self.history]
        mean_scores = [row["mean_score"] for row in self.history]
        std_scores  = [row["std_score"] for row in self.history]
        pos_mean    = [row["best_position_mean"] for row in self.history]
        pos_std     = [row["best_position_std"] for row in self.history]

        self.line1.set_data(iterations, best_scores)
        self.line2.set_data(iterations, mean_scores)
        self.line3.set_data(iterations, std_scores)
        self.line4.set_data(iterations, pos_mean)
        self.line5.set_data(iterations, pos_std)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    # Plot Final Optimization Results (must be called explicitly)
    def plot(self):
        iterations  = [row["iteration"]             for row in self.history]
        best_scores = [row["best_score"]            for row in self.history]
        mean_scores = [row["mean_score"]            for row in self.history]
        std_scores  = [row["std_score"]             for row in self.history]
        pos_mean    = [row["best_position_mean"]    for row in self.history]
        pos_std     = [row["best_position_std"]     for row in self.history]

        fig = plt.figure(figsize=(6, 3))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1,  1.2])
        
        # Best Score (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, best_scores, color='blue', linewidth=2)
        ax1.set_ylabel("Best Score")
        ax1.set_title("Best Score")
        ax1.grid(True)

        # Mean Score (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations, mean_scores, color='green', linestyle='--')
        ax2.set_ylabel("Mean Score")
        ax2.set_title("Mean Score")
        ax2.grid(True)

        # Std Score (second row, full width)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iterations, std_scores, color='orange')
        ax3.set_ylabel("Std Score")
        ax3.set_title("Fitness Standard Deviation")
        ax3.grid(True)

        # Best Position Mean and Std (third row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(iterations, pos_mean, label="Best Position Mean")
        ax4.plot(iterations, pos_std, label="Best Position Std", linestyle='--')
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Position Stats")
        ax4.set_title("Best Position Statistics")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def save(self) -> None:
        """
        Save the optimization history to both JSON and CSV files.

        - JSON includes the full history in structured format.
        - CSV includes selected fields for easier inspection and analysis.
        """
        # Save to JSON
        with open(self.json_path, "w", encoding="utf-8") as f_json:
            json.dump(self.history, f_json, indent=2)

        # Save to CSV
        fieldnames = [
            "iteration",
            "best_score",
            "mean_score",
            "std_score",
            "best_position",
            "best_position_mean",
            "best_position_std",
        ]

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow({
                    "iteration": row.get("iteration"),
                    "best_score": row.get("best_score"),
                    "mean_score": row.get("mean_score"),
                    "std_score": row.get("std_score"),
                    "best_position": row.get("best_position"),
                    "best_position_mean": row.get("best_position_mean"),
                    "best_position_std": row.get("best_position_std"),
                })

    def run(self, mode="Inertia",c1=2.0, c2=2.0,inertia=0.7, w_min=0.4, w_max=0.9):
        
        if self.verbose == 1:
            print("=================================================")
            print("Starting APSO optimization .")
            
        self.c1      = c1
        self.c2      = c2
        self.w_min   = w_min
        self.w_max   = w_max
        self.inertia = inertia
        
        if mode.lower()     == "inertia":
            return self.IntertiaOptimizer()
        elif mode.lower()   == "avg":
            return self.Optimizer()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'inertia' or 'avg'.")
    
    # Adaptive Inertia Optimizer
    def IntertiaOptimizer(self):
    
        positions               = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities              = np.zeros((self.num_particles, self.dim))
        personal_best           = positions.copy()
        personal_best_scores    = self.evaluate_population(personal_best, self.objective_function)
        global_best_idx         = np.argmin(personal_best_scores)
        global_best             = personal_best[global_best_idx].copy()
        global_best_score       = personal_best_scores[global_best_idx]

        self._OP_started()

        for iteration in tqdm(range(self.max_iter), desc="Optimizing ", ncols=80):
        
            # Initialize inertia weights per particle
            inertia_weight = self.inertia * (1 - iteration / self.max_iter)

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            
            # Compute new velocities and positions
            velocities = (
                inertia_weight * velocities
                + self.c1 * r1 * (personal_best - positions)
                + self.c2  * r2 * (global_best - positions)
            )
            
            positions                       += velocities
            positions                       = np.clip(positions, self.lb, self.ub)

            fitness                         = self.evaluate_population(positions, self.objective_function)

            improved                        = fitness < personal_best_scores
            personal_best[improved]         = positions[improved]
            personal_best_scores[improved]  = fitness[improved]

            if np.min(fitness) < global_best_score:
                global_best_score   = np.min(fitness)
                global_best         = positions[np.argmin(fitness)].copy()
            
            # Save history
            self._OP_write(iteration,global_best_score,fitness,global_best)
            
        self._OP_end(global_best_score)
         
        return global_best, global_best_score, self.history, self 
    
    # Adaptive weights Optimizer
    def Optimizer(self):
        
        positions               = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities              = np.zeros_like(positions)
        personal_best           = positions.copy()
        personal_best_scores    = self.evaluate_population(personal_best,self.objective_function)
        global_best_idx         = np.argmin(personal_best_scores)
        global_best             = personal_best[global_best_idx].copy()
        global_best_score       = personal_best_scores[global_best_idx]

        # Initialize adaptive weights per particle
        inertia_weights         = np.full(self.num_particles, self.w_max)
        
        self._OP_started()

        for iteration in tqdm(range(self.max_iter), desc="Optimizing ", ncols=80):
            # Compute new velocities and positions
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            velocities  = (
                inertia_weights[:, np.newaxis] * velocities +
                self.c1 * r1 * (personal_best - positions) +
                self.c2 * r2 * (global_best - positions)
            )
            positions   = np.clip(positions + velocities, self.lb, self.ub)

            # Evaluate
            fitness     = self.evaluate_population(positions,self.objective_function)

            # Update personal and global bests
            improved                        = fitness < personal_best_scores
            personal_best[improved]         = positions[improved]
            personal_best_scores[improved]  = fitness[improved]

            if np.min(fitness) < global_best_score:
                global_best_score           = np.min(fitness)
                global_best                 = positions[np.argmin(fitness)].copy()

            # Update inertia weights (adaptive)
            F_bar           = np.mean(fitness)
            better_than_avg = fitness <= F_bar
            inertia_weights[better_than_avg] = self.w_min + (
                (fitness[better_than_avg] - global_best_score) * (self.w_max - self.w_min)
            ) / (F_bar - global_best_score + 1e-10)  # avoid div by 0
            inertia_weights[~better_than_avg] = self.w_max
            
            # Save history
            self._OP_write(iteration,global_best_score,fitness,global_best)

        self._OP_end(global_best_score)
        
        return global_best, global_best_score, self.history, self
     

    # common constructor    
    def _OP_started(self):
    
        self.history        = []
        self.start_time     = time()
        
        if self.verbose == 1: 
            print("Firing APSO optimization ..")
    
    # common writer
    def _OP_write(self,iteration,global_best_score,fitness,global_best):
    
        self.history.append({
            "iteration"         : iteration + 1,
            "best_score"        : float(global_best_score),
            "mean_score"        : float(np.mean(fitness)),
            "std_score"         : float(np.std(fitness)),
            "best_position"     : global_best.tolist(),
            "best_position_mean": float(np.mean(global_best.tolist())),
            "best_position_std" : float(np.std(global_best.tolist())),
        })
            
        if self.live_plot:
            self._update_plot()

        formatted_position = ",".join(f"{x:.3f}" for x in global_best.tolist())
        tqdm.write(f"Iter {iteration+1}/{self.max_iter}, Best Position: {formatted_position}, Best Score: {global_best_score:.4f}, Mean Score: {np.mean(fitness):.4f}")
   
    # common destructor
    def _OP_end(self,global_best_score):
    
        duration = time() - self.start_time
        
        if self.verbose == 1:
            print(f"Optimization completed in {duration:.2f} seconds.")
            print(f"Best Score:               {global_best_score:.4f}")
            print("APSO optimization Completed ...")
            print("=================================================")
        
        if self.live_plot:
            plt.ioff()
            plt.show()
    