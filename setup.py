from setuptools import setup, find_packages

setup(
    name='pyapso',
    version='1.0.5',  
    description='Adaptive Particle Swarm Optimization for continuous, multi-objective, and multidimensional problems',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/smohamadabedy/pyapso',  # replace with your GitHub
    project_urls={
        "Documentation": "https://github.com/smohamadabedy/pyapso#readme",
        "Source": "https://github.com/smohamadabedy/pyapso",
        "Issues": "https://github.com/smohamadabedy/pyapso/issues",
    },
    license='MIT',
    keywords='optimization particle-swarm metaheuristic multiobjective',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib'
    ],
    include_package_data=True,
)
