# custom_gates.py
from qiskit import QuantumCircuit,QuantumCircuit,QuantumRegister
from qiskit.circuit import Gate
import numpy as np
import toolbox
import warnings


def f_swap_chain(i,j) :
    """
    Apply a chain of Fermionic swap over adjacent bits, with some optimization already included

    Parameters:
    - i (int) : one of the qubits to fermionic swap
    - j (int) : one of the qubits to fermionic swap

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate
    """
    a=int(min(i,j))
    b=int(max(i,j))
    qc = QuantumCircuit(b-a+1, name="Fswap")
    for k in range(1,b-a) :
        qc.cz(0,k)
        qc.cz(b-a,b-a-k)
    qc.swap(0,b-a)
    qc.cz(0,b-a)
    return qc.to_gate()
    
def F(k=0,n=0):
    """
    Apply the two-qubit fermionic Fourier transform gate F_{k,n} in the n-mode FFFT
    The bits are reversed in order to obtain the expected operator form E11 in http://arxiv.org/abs/2012.09238
    Source : Figure 8 from http://arxiv.org/abs/1902.10673

    Parameters:
    - k (int) : appears in the initial rotation Rz(2*pi*k/n)
    - n (int) : n-mode FFFT, appears in the initial rotation Rz(2*pi*k/n)

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate
    """
    qc = QuantumCircuit(2, name="F")

    if k!=0 and n!=0 :
        qc.rz(2*np.pi*k/n, 1)
    qc.s(0)
    qc.h([1,0])

    qc.cx(1, 0)

    qc.h(1)
    qc.s(1)
    qc.tdg(0)

    qc.cx(1, 0)

    qc.h(1)
    qc.t(0)

    qc.cx(1,0)

    qc.h(1)
    qc.s(1)
    qc.h(0)

    # Convert the circuit to a gate
    return qc.to_gate()

def hopping_plaquette(tau,t,k=0,n=0,controlled=False):
    """
    Apply hopping plaquette for a single plaquette (4 sites)
    Source : Appendix E from http://arxiv.org/abs/2012.09238

    Parameters:
    - tau (double) : interaction strengh of the hopping term.
    - t (double) : time for the time evolution.
    - k (int, optional) : parameter for FFFT. Default is 0.
    - n (int, optional) : parameter for FFFT. Default is 0.
    - controlled (bool, optional): Define if the gate is controlled or not. The control register needs to be the last given in the range of bits to append the gate to. Default is False

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate.
    """
    QR = QuantumRegister(4, "QR")
    PEA = QuantumRegister(1, "PEA") if controlled else None

    registers = [QR]
    if controlled and PEA:
        registers.append(PEA)
    qc = QuantumCircuit(*registers, name="Single plaquette")
    # For naming we need to define the gate before appending
    F_3_1=F(k=k,n=n)
    F_3_1.name=(r'F_{3,1}')
    F_2_4=F(k=k,n=n)
    F_2_4.name=(r'F_{2,4}')

    # The F gates with the needed fermionic swap operator. Some simplification are already done to cancel out double swaps and double  controlled Z rotation
    qc.swap(QR[1],QR[2])
    qc.cz(QR[1],QR[2])
    qc.append(F_3_1,[QR[1],QR[0]])
    qc.append(F_2_4,[QR[2],QR[3]])
    qc.cz(QR[1],QR[2])
    qc.swap(QR[1],QR[2])

    # If we do the "Using a Clifford we can rotate X ⊗ X and Y ⊗ Y to Z ⊗ I and I ⊗ Z"
    # Y ⊗ Y part
    # Transformation Y to Z
    qc.sdg(QR[1])
    qc.sdg(QR[2])
    qc.h(QR[1])
    qc.h(QR[2])
    qc.cx(QR[2],QR[1])
    # Rotation
    if controlled :
        qc.cx(PEA[0],QR[1])
        qc.rz(2*t*tau, QR[1])
        qc.cx(PEA[0],QR[1])
    else :
        qc.rz(-2*t*tau, QR[1])  # Z rotation for a2
    # Transformation Z to Y
    qc.cx(QR[2],QR[1])
    qc.h(QR[2])
    qc.h(QR[1])
    qc.s(QR[2])
    qc.s(QR[1])
    # X ⊗ X part
    # Transformation X to Z
    qc.h(QR[1])
    qc.h(QR[2])
    qc.cx(QR[1],QR[2])
    # Rotation
    if controlled :
        qc.cx(PEA[0],QR[2])
        qc.rz(2*t*tau, QR[2]) 
        qc.cx(PEA[0],QR[2])
    else :
        qc.rz(-2*t*tau, QR[2])  # Z rotation for a3
    # Transformation Z to X
    qc.cx(QR[1],QR[2])
    qc.h(QR[2])
    qc.h(QR[1])

    # Inversing of the F gates
    qc.swap(QR[1],QR[2])
    qc.cz(QR[1],QR[2])
    qc.append(F_2_4,[QR[2],QR[3]])
    qc.append(F_3_1,[QR[1],QR[0]])
    qc.cz(QR[1],QR[2])
    qc.swap(QR[1],QR[2])

    return qc.to_gate()


def hopping_plaquettes(L,tau,t,color,k=0,n=0,controlled=False,debug=False) :
    """
    Apply hopping plaquette for all the plaquettes independantly in the whole lattice.
    We assume periodic boundary condition such that each edge belong to a single plaquette color.
    The indices are mapped row wise, with qbits 0 to L^2-1, and considers only a single spin.
    Qubit 0 is the top left site, on the top left of a pink plaquette, which is also the bottom right of a gold plaquette from periodic effect. As Figure 1 in the source.
    Source : Appendix E from http://arxiv.org/abs/2012.09238

    Parameters:
    - L (int) : must be even, side length of the lattice. 
    - tau (double) : interaction strengh of the hopping term.
    - t (double) : time for the time evolution.
    - color (str) : pink or gold. Is that applying to the pink or the gold plaquettes. Used for indexing.
    - k (int, optional) : parameter for FFFT. Default is 0.
    - n (int, optional) : parameter for FFFT. Default is 0.
    - controlled (bool, optional): Define if the gate is controlled or not. The control register needs to be the last given in the range of bits to append the gate to. Default is False
    - debug (bool, optional) : return or not the list of plaquettes for debugging purpose. Default is False

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate.
    """
    if color not in ["pink","gold"]:
            raise ValueError("The color of the plaquette must be either pink or gold")
    if L%2==1:
            raise ValueError("L must be even")
    
    QR = QuantumRegister(L**2, "QR")
    PEA = QuantumRegister(1, "PEA") if controlled else None

    registers = [QR]
    if controlled and PEA:
        registers.append(PEA)
    qc = QuantumCircuit(*registers, name=f'H_h,{color[0]}')

    plaquettes = []
    match color :
        case "pink" : # For pink there are no periodic boundary effect 
            i = 0
            while i <= L**2-L-2:
                plaquettes.append(i)
                plaquettes.append(i+1)
                plaquettes.append(i+L+1)
                plaquettes.append(i+L)
                i+=2
                if (i+L)%(2*L)==0 :
                    i+=L
        case "gold" : # Periodic boundary condition to consider
            i=L+1
            while i <= L**2-1:
                # If it's the bottom right corner
                if i == L**2-1:
                    plaquettes.append(i)
                    plaquettes.append(i-L+1)
                    plaquettes.append(0)
                    plaquettes.append(L-1)
                # If on the right side
                elif (i+1)%(L)==0 :
                    # One spin
                    plaquettes.append(i)
                    plaquettes.append(i-L+1)
                    plaquettes.append(i+1)
                    plaquettes.append(i+L)
                # If it's on the bottom side
                elif i>L**2-L :
                    plaquettes.append(i)
                    plaquettes.append(i+1)
                    plaquettes.append(i%(L)+1)
                    plaquettes.append(i%(L))
                # If it's normal center and simple
                else :
                    plaquettes.append(i)
                    plaquettes.append(i+1)
                    plaquettes.append(i+L+1)
                    plaquettes.append(i+L)
                # If we reach the end of a line
                if (i+1)%(L) ==0 :
                    i+=L # Skip a line
                i+=2
    swap =toolbox.get_swap_sequence(list(range(L**2)),plaquettes)
    for (i,j) in swap :
        qubits = [QR[k] for k in range(i, j + 1)]
        qc.append(f_swap_chain(i,j),qubits)
    for i in range(int((L**2)/4)) :
        qc.append(hopping_plaquette(tau=tau,t=t,controlled=controlled),[QR[4*i],QR[4*i+1],QR[4*i+2],QR[4*i+3]]+([PEA[0]] if controlled else []))
    # Reverse all the Fswap
    for (i,j) in reversed(swap) :
        qubits = [QR[k] for k in range(i, j + 1)]
        qc.append(f_swap_chain(i,j),qubits)
    if debug == True :
        return qc.to_gate(), plaquettes
    else :
        return qc.to_gate()

def hopping_tile(L,tau,t,step=1,k=0,n=0,controlled=False) :
    """
    Apply hopping plaquette for the full lattice
    Each hopping tile implement H = [e^(i(tau/(2*steps))H_p)e^(i(tau)H_g/steps)e^(i(tau/(2*steps)H_p)]^(steps) with some optimization so reduce the number of application.
    So it is 3 application of the hopping plaquettes, twice for the pink one and one for the gold ones.
    We assume periodic boundary condition such that each edge belong to a single plaquette color.
    Source : Appendix E from http://arxiv.org/abs/2012.09238

    Parameters:
    - L (int) : must be even, side length of the lattice. 
    - tau (double) : interaction strengh of the hopping term.
    - t (double) : time for the time evolution.
    - step (int) : number of Trotter steps to do. Default is 1.
    - k (int, optional) : parameter for FFFT. Default is 0. 
    - n (int, optional) : parameter for FFFT. Default is 0.
    - controlled (bool, optional): Define if the gate is controlled or not. The control register needs to be the last given in the range of bits to append the gate to. Default is False

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate.
    """
    QRup = QuantumRegister(L**2, "QR1")
    QRdown = QuantumRegister(L**2, "Q2")
    PEA = QuantumRegister(1, "PEA") if controlled else None

    registers = [QRup,QRdown]
    if controlled and PEA:
        registers.append(PEA)
    qc = QuantumCircuit(*registers, name="H_h")
    # Applies it first on spin groupe and then on the other one
    for qbit_list in [[*QRup]+([PEA[0]] if controlled else []),[*QRdown]+([PEA[0]] if controlled else [])] :
    # In the case where L=2, there is only a single plaquette to consider, gold plaquette would double the interaction between the same terms. And so the number of step is also irrelevant.
        if L==2 :
            qc.append(hopping_plaquettes(L,tau,t,"pink",controlled=controlled), qbit_list)
        else :
            qc.append(hopping_plaquettes(L,tau,t/(2*step),"pink",controlled=controlled), qbit_list)
            qc.append(hopping_plaquettes(L,tau,t/step,"gold", controlled=controlled), qbit_list)
            for _ in range(step-1) :
                qc.append(hopping_plaquettes(L,tau,t/step,"pink", controlled=controlled), qbit_list)
                qc.append(hopping_plaquettes(L,tau,t/step,"gold", controlled=controlled), qbit_list) 
            qc.append(hopping_plaquettes(L,tau,t/(2*step),"pink", controlled=controlled), qbit_list)

    return qc.to_gate()

def interaction(L, u, t, controlled=False):
    """
    Apply the interaction Hamiltonian H_I = (u/4) * sum_i z_i_up z_i_down
    using controlled Rz gates to simulate the time evolution operator e^(-i*t*H_I).
    Source : Equation 6 from https://arxiv.org/pdf/2012.09238
    
    Parameters:
    - L (int) : must be even, side length of the lattice. 
    - u (double) : interaction strength for the interaction term.
    - t (double) : time for the time evolution.
    - controlled (bool, optional): Define if the gate is controlled or not. The control register needs to be the last given in the range of bits to append the gate to. Default is False

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate.
    """

    QRup = QuantumRegister(L**2, "QR1")
    QRdown = QuantumRegister(L**2, "Q2")
    PEA = QuantumRegister(1, "PEA") if controlled else None

    registers = [QRup,QRdown]
    if controlled and PEA:
        registers.append(PEA)
    qc = QuantumCircuit(*registers, name="H_I")
    theta = u * t / 4  # Angle for the Rz rotation based on interaction strength and time
    for i in range(L**2):
        qc.cx(QRup[i], QRdown[i])

        if controlled :
            qc.cx(PEA[0], QRdown[i])
            qc.rz(2 * theta, QRdown[i])
            qc.cx(PEA[0], QRdown[i])
        else :
            qc.rz(-2 * theta, QRdown[i])  # Apply Rz to the second qubit (down)
        qc.cx(QRup[i], QRdown[i])

    return qc.to_gate()

def hubbard_unitary(L,tau,u,t,step_plaq=1,trotter_step=1,r=1,k=0,n=0,controlled=False) :
    """
    Implemente the unitary for the Hubbard model as shown the source.
    The two trotterization (plaquette and overall) were added to the equation E1 and optimized based on the source.
    The final version seen here comes from further developpement shown in the report. 
    Source : Appendix E, equation (E1), from http://arxiv.org/abs/2012.09238

    Parameters:
    - L (int) : must be even, side length of the lattice. 
    - tau (double) : interaction strengh of the hopping term.
    - u (double) : interaction strength for the interaction term.
    - t (double) : total time for the time evolution.
    - step_plaq (int) : number of Trotter steps to do in the decomposition of the plaquette operation. Default is 1.
    - trotter_step (int) : number of Trotter steps to do in the decomposition of the plaquette operation. Default is 1.
    - r (int) : number of repetition of the unitary that need to be done. Default is 1 meaning a total of 1 unitary.
    - k (int, optional) : parameter for FFFT. Default is 0. 
    - n (int, optional) : parameter for FFFT. Default is 0.
    - controlled (bool, optional): Define if the gate is controlled or not. The control register needs to be the last given in the range of bits to append the gate to. Default is False

    Returns:
    - qc.to_gate() : the quantum circuit as a custom gate.
    """
    QRup = QuantumRegister(L**2, "QR1")
    QRdown = QuantumRegister(L**2, "Q2")
    PEA = QuantumRegister(1, "PEA") if controlled else None

    registers = [QRup,QRdown]
    if controlled and PEA:
        registers.append(PEA)
    qc = QuantumCircuit(*registers, name="U" )
    qbit_list= [*QRup,*QRdown]+([PEA[0]] if controlled else [])

    qc.append(interaction(L,u,t/(2*trotter_step), controlled=controlled),qbit_list)
    # Here we need to have time as t/trotter_step so that the exponent is still correct but every term is devided by trotter_step*step_plaq
    qc.append(hopping_tile(L,tau,t/trotter_step,step=step_plaq,k=0,n=0,controlled=controlled),qbit_list)
    for _ in range(int(trotter_step*r)-1) :
        qc.append(interaction(L,u,t/trotter_step, controlled=controlled),qbit_list)
        qc.append(hopping_tile(L,tau,t/trotter_step,step=step_plaq,k=0,n=0,controlled=controlled),qbit_list)
    qc.append(interaction(L,u,t/(2*trotter_step), controlled=controlled),qbit_list)

    return qc.to_gate()
