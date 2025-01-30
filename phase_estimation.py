# phase_estimation.py
from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister,transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import numpy as np
import custom_gate


def single_phase_estimation_run(L,tau,u,t,precision,state,step_plaq=1,trotter_step=1,shots=100,backend=AerSimulator(method="statevector")):
    """
    Perform a phase estimation and returns the counts in order to be modifed and extract the phase.

    Parameters:
    - L (int) : must be even, side length of the lattice. 
    - tau (double) : interaction strengh of the hopping term.
    - u (double) : interaction strength for the interaction term.
    - t (double) : total time for the time evolution.
    - precision (int) : the length of the register used for phase estimation, also linked with overall precision and number of controlled unitary ran.
    - state (list) : the descrition of the state that will initialize the system.
    - step_plaq (int) : number of trotter steps for the decomposition of the hopping terms. Default is 1.
    - trotter_step (int) : number of trotter steps for the overall decomposition. Default is 1.
    - shots (int) : number of shots in the simulation. Default is 100.
    - backend (AerSimulator, optional): The backend used to execute the circuit. Default is AerSimulator with the "statevector"  method. This gives the easiest result for tests. 
                                        But very limited in size. Prefer "matrix_product_state" since it allows circuit larger than 30 qubit.  

    Returns:
    - counts (dict) : the resulting counts of the phase estimation algorithm.
    """
    PEA = QuantumRegister(precision, "PEA")
    QR = QuantumRegister(2 * L**2, "QR")
    cr = ClassicalRegister(precision, "c")
    qc = QuantumCircuit(QR, PEA, cr)
    
    # State preparation if provided
    if len(state) > 0:
        qc.initialize(state, [*QR])
    
    qc.h([*PEA])
    for p in range(precision):
        qc.append(custom_gate.hubbard_unitary(L, tau, u, t/2, step_plaq=step_plaq, trotter_step=trotter_step, r=2**p, k=0, n=0, controlled=True), [*QR, PEA[p]])
    
    qc.append(QFT(precision, inverse=True, do_swaps=True), [*PEA])
    qc.measure([*PEA], [*cr])
    
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts(qc)

    return counts