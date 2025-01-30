# Quantum implementation and simulation of the Hubbard model
 Master project from Gilles-Henry Moreillon in Quantum Science and Engineering at EPFL (2025).


 Under supervision of Professor Giovanni De Micheli form the Integrated Systems Laboratory and Doctor Mathias Soeken
 

This repository contains the implementation of a quantum simulation for the Hubbard model using **plaquette trotterization**. The project uses Python and **Qiskit** to simulate the time evolution of the Hubbard model.


## Project Structure

- `custom_gate.py`: Contains the custom gates implemented for the project. These gates are used to optimize the quantum circuits for simulating the Hubbard model.
- `figures.ipynb`: Jupyter notebook that generates all the figures for the report. This includes visualizations and graphs to support the findings of the project.
- `phase_estimation.py`: Contains the function for quantum phase estimation, which is used to retrieve phase and energy eigenvalues during the simulation.
- `toolbox.py`: Utility functions that provide small but useful operations to support the main algorithms.
- `environment.yml`: Defines the required dependencies and environment setup for running the project.
- `Report.pdf`: The final report that outlines the project’s methodology, results, and conclusions.

## Getting Started

To get started with this project, follow the instructions below.

### 1. Clone the Repository

```bash
git clone https://github.com/GHMoreillon/QuantumSimulationHubbard.git
cd QuantumSimulationHubbard
```
### 2. Set Up the Environment
To set up the project environment, use the environment.yml file with Conda:

```bash
conda env create -f environment.yml
conda activate hubbard-simulation
```
This will create a Conda environment with all the necessary dependencies.


## License
This project is licensed under the MIT License - see the LICENSE file for details.