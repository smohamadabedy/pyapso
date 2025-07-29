from setuptools import setup, find_packages

setup(
    name='PyApso',
    version='1.0.0',
    description='Adaptive Particle Swarm Optimization for continuous, multi-objective, and multidimensional problems',
    author='SMH Abedy',
    author_email='m.h.o.abedu@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tqdm',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
