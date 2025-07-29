# PyApso

**PyApso** is an implementation of the **Adaptive Particle Swarm Optimization (APSO)** algorithm and designed for solving continuous, multi-objective, and multi-dimensional optimization problems. 

---

## Installation

Install the package using pip:

```bash
pip install pyapso
````
Or from source:
```bash
git clone https://github.com/smohamadabedy/pyapso.git
cd pyapso
pip install .
```

## Features
ADAPTIVE -  key parameters: (inertia weight, avg weight)
Continuous optimization support
Single and multi-objective fitness evaluation
Constraint-aware evolution
Batch evaluation and parallel execution
Excel and JSON logging
Custom callbacks and visualizations

## Usage

```python
from pyapso import APSO
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
```
## Modes

```python
apso.run("avg", c1=1.8, c2=1.8, w_min=0.5, w_max=0.9)
```
"avg" mode adapts the inertia weight dynamically between w_min and w_max using swarm behavior.

```python
apso.run("inertia",c1=1.8, c2=1.8, inertia=0.05)
```
"inertia" mode adapts a fixed inertia weight throughout the optimization (default).

💡 Tip:

 - A higher c1 encourages particles to explore their own search path.
 - A higher c2 encourages convergence to the global best.
 - Lower inertia means the swarm is more reactive and less explorative.



## Examples
You can find runnable demos in the examples/ directory:

```bash
examples/
├── demo_1.py    # 5D Rastrigin function
├── demo_2.py    # 3D constrained benchmark
├── demo_3.py    # 2D McCormick function
```

Run an example:
```bash
python examples/demo_2.py
```
