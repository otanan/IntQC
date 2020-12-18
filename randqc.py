#!/usr/bin/env python3
"""
====================================
Filename:       randqc.py
Authors:        Jonathan Delgado, Joseph Farah, Valentino Crespi
Date created:   November 2019
Description:    Generate random quantum circuits and produce images of quantum
                states operated on by these circuits for later analysis.
====================================
"""
######################## Imports ########################
import os, sys

#------------- Math imports -------------#
import random
import numpy as np, scipy
import qutip as qt

# for info file parsing of dictionary state_strings
import ast
from datetime import datetime
import argparse

from progress.bar import ShadyBar

#------------- Custom scripts -------------#
import qstate_imager

# The topology of the qubits, if false all qubits can be connected
    # if true, only adjacent qubits can be placed in the same gate
_ring_topology = False

def _parse_args():
    """
        Introduces the command-line arguments allowed and does basic error
        checking on these arguments.
    
        Args:
            none (none): none
    
        Returns:
            none (none): none
    
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #------------- Available command-line arguments -------------#

    # Number of qubits in the system
    parser.add_argument(
        '--Nqbits', '-N', type=int, default=4,
        help='Number of qubits.'
    )

    # Sampling
    parser.add_argument(
        '--nsamples', '-n', type=int, default=1,
        help='Number of trials to do with this set of instructions.'
    )

    parser.add_argument(
        '--bipartition', '-bi', type=int, default=-1,
        help='Where to place the bipartition. If negative, defaults to splitting the state in half. Since the qubits are zero-indexed, all qubits with an index lower than the bipartition will be placed in the first partition. Provide a bipartition of 0 to calculate the bipartition average for the entropy data. However images cannot be gathered when the bipartition is 0.'
    )

    # Instruction string argument used to determine the amount of circuits
        # to use and the amount and types of gates to use in each circuit
    parser.add_argument(
        '--instructions', '-inst', type=str, default='3*(Clx8)',
        help='Formatting to construct a circuit with arbitrary sub-circuit pieces with arbitrary gate sizes. An example string is:\n \'4*(Clx12,Tx4);Ux10;Clx40\'. The n* indicates n layers of the subsequent circuit instructed. Blocks are separated by semicolon while sub-blocks in a layer are separated by a comma.'
    )

    parser.add_argument(
        '--local', '-local', action='store_true', default=False,
        help='Controls the topology of the qubits. If True gates must be applied only to adjacent qubits.'
    )

    parser.add_argument(
        '--input_state', '-q0', type=str, default='zero',
        help='Option to control the input state for the random quantum circuit. The options are: \'zero\' for the zero state, \'prod\' for a random product state, and \'rand\' for a completely random quantum state.'
    )

    #------------- Printing & Saving -------------#
    # Saving image after each circuit is applied
    parser.add_argument(
        "--no_save_post_circuit", "-nspc", action='store_true', default=False,
        help='Option to not save images of state after being applied to each circuit.'
    )

    parser.add_argument(
        "--colored", "-color", action='store_true', default=False,
        help='Option for images generated post circuit to be colored.'
    )

    # Generate training data
    parser.add_argument(
        "--training", "-train", action='store_true', default=False,
        help='Option to have the simulation prepare the outputs as training data. Affects output location and file saving.'
    )

    parser.add_argument(
        '--entropy', '-H', type=float, default=-1,
        help='Determines which Renyi entropy is calculated and saved after each gate is applied. An argument of 1 will calculate and save the Von-Neumann entropy. Provide a negative number to not save entropy data.'
    )

    parser.add_argument(
        "--save_state", "-saveq", action='store_true', default=False,
        help='Saves the output state in a readable format for Qutip.'
    )

    # Save nothing
    parser.add_argument(
        "--no_save", "-ns", action='store_true', default=False,
        help='Option to remove any file saving features. Helpful for quick testing.'
    )

    # Save everything: such as training data, entropy, images post circuit
    parser.add_argument(
        "--save_all", "-save", action='store_true', default=False,
        help='Option to save everything such as training data, images post circuit applications, and entropy data.'
    )    

    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='Print also the matrix to the standard output'
    )

    parser.add_argument(
        "--output_path", "-o", type=str, default='results',
        help="Output directory for any file saving."
    )

    args = parser.parse_args()

    #------------- Parsing of arguments and error checking -------------#

    if args.save_all and args.no_save:
        print('Contradictory saving instructions. Unsure to save everything or nothing.')
        print('Exiting...')
        sys.exit()        

    args.save_flags = {
        'save_post_circuit': not args.no_save_post_circuit,
        'training': args.training,
        'entropy': args.entropy >= 0,
        'save_state': args.save_state,
    }

    # Change the flags based on saving everything or nothing
    if args.save_all or args.no_save:
        for flag in args.save_flags:
            args.save_flags[flag] = True if args.save_all else False

    return args

def _prepare_saving(N, q0, instructions_string, bipartition, type_of_circuit, save_path, save_flags):
    """
        Handles folder generation for saving outputs of simulations and the naming of each file.

    """
    root = save_path
    save_path = get_save_path(N, q0, instructions_string, bipartition, _ring_topology, root=root)

    # We don't need to do anything if the path already exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #------------- Folder generation -------------#

    # Extra folder containing all training data
    if save_flags['training']:
            # Example training data folder paths
                # training_data-N(12)-q0(rand)-inst[circuit]-Universal.jpeg
                # training_data-N(12)-q0(prod)-inst[circuit]-Clifford.jpeg
        training_data_path = get_training_data_path(N, q0, instructions_string, bipartition, type_of_circuit, local_top=_ring_topology, root=root)

        if not os.path.exists(training_data_path):
            os.mkdir(training_data_path)

    # Folder for holding qutip files of output states
    if save_flags['save_state']:
        save_state_path = get_save_state_path(N, q0, instructions_string, bipartition, type_of_circuit, local_top=_ring_topology, root=root)

        if not os.path.exists(save_state_path):
            os.mkdir(save_state_path)

def parse_instructions(instructions, delimiter=';'):
    """
        Helper function that parses input string for the amount of circuits to use, the amount of gates, and the types at each point. An example valid input string is: 4*(Clx8,Tx12);Clx40;Ux10. The n* indicates n layers of the subsequent circuit instructed.

        Args:
            instructions (string): the instructions themselves.
            delimiter (string): how different circuit pieces are separated
    
        Returns:
            instructions (list): a list of 2-tuples containing first the gate 
            type and second the amount of gates to use
    
    """
    # The actual instructions to later be interpreted by the circuit generator
    circuit_instructions = []
    # If no instructions are passed we simply want to run an empty circuit.
        # i.e. an identity operation on our input state
    if instructions == '':
        return circuit_instructions

    # Each piece is separated by a semicolon or comma depending on the
        # nesting. The delimiter command is used for recursively parsing
        # the nested circuit.
    instructions = instructions.split(delimiter)

    for instruction in instructions:
        # Split any repeated blocks into the sub-block and its amount of times to be repeated
        instruction = instruction.split('*')

        # If true, then the instruction is being asked to be repeated
        if len(instruction) == 2:
            # Remove the parentheses, append a comma to the end of the repeated
                # command. Repeat it, then remove the final additional comma
            instruction = (int(instruction[0]) * (instruction[1][1:-1] + ','))[:-1]

            # Recursively call the function to now parse the instructions
                # for this repeated block and then append it to our
                # current list of instructions
            circuit_instructions += parse_instructions(instruction, delimiter=',')

            continue

        # Separates each circuit into (type, gate_amounts)
        pair = instruction[0].split('x')
        # Corrects any capitalization. Important to be done here
            # since this will go into the filename
        pair[0] = pair[0].capitalize()

        try:
            gate_count = int(pair[1])
        except ValueError:
            print(f'Invalid instructions. Expected integer for gate count. Received "{pair[1]}" instead.')
            raise

        circuit_instructions.append((pair[0], gate_count))

    return circuit_instructions

def parse_state_string(N, state_string):
    """
        Helper function that returns a quantum state depending on the string 
        passed in. Example valid state_strings are 'rand' for a random Haar 
        state, or 'prod' for a random product state.
    
        Args:
            state_string (string): the name of the state itself. Examples include: zero, prod, rand
    
        Returns:
            q (qutip.qObj): the quantum state
    
    """
    # Fix any capitalization
    state_string = state_string.lower()

    # Selects the input state based on the command
    if state_string == 'zero':
        return zero_state(N)
    
    if state_string == 'prod':
        return rand_prod_state(N)
    
    if state_string == 'phase':
        return phase_state(N)

    if state_string == 'rand':
        return rand_state(N)

    if state_string == 'exp_rand':
        return _experimental_rand_state(N)

    if state_string == 'exp_prod':
        return _experimental_rand_prod_state(N)
    
    # No state was found to return
    print('Invalid input state. Read --help to see valid input state commands.')
    print('Exiting...')
    sys.exit()

######################## Quantum states ########################
def zero_state(N):
    """
        Build N tensor product of |0>
    
        Args:
            N (int): the number of qubits in the system this state will
            represent.
    
        Returns:
            q (qutip.qObj): the state
    
    """
    return qt.qip.qubits.qubit_states(N, states=[0])

#------------- Random states -------------#
def rand_prod_state(N):
    """
        Generates a random Haar product state.
    
        Args:
            N (int): the number of qubits in the system this state will
            represent.
    
        Returns:
            q (qutip.qObj): the state
    
    """
    return qt.tensor( [qt.rand_ket_haar(2) for _ in range(N)] )

def _experimental_rand_prod_state(N):
    """
        Generates a random Haar product state.
    
        Args:
            N (int): the number of qubits in the system this state will
            represent.
    
        Returns:
            q (qutip.qObj): the state
    
    """
    return qt.tensor( [rand_state(1) for _ in range(N)] )

def rand_state(N):
    """
        Generates a random Haar state.
    
        Args:
            N (int): the number of qubits in the system this state will
            represent.
    
        Returns:
            q (qutip.qObj): the state    
    """

    # Ensures the dimensions of the random Haar state match the dimensions of
        # other states    
    return qt.rand_ket_haar(2**N, dims=[[2] * N, [1] * N])

def phase_state(N, phases=[]):
    """
        Creates a phase state of the form: (cos(𝜃1) |0> + sin(𝜃1) |1>) ⊗ … ⊗ (cos(𝜃n) |0> + sin(𝜃n) |1>) with 𝜃 ∈ [0,𝜋]
    
        Args:
            N (int): the number of qubits in the system
            phases (list): a list of the phases to use, leave as default argument to generate a random phase state.
    
        Returns:
            q (qutip.Qobj): the phase state
    
    """
    if len(phases) > 0 and len(phases) < N:
        print('Insufficient phases to generate a phase state. Returning a random phase state.')
        phases = []

    cosphases = np.cos(
            (np.pi * np.random.default_rng().random(N))
            if len(phases) == 0 else phases
        )
    return qt.qip.qubits.qubit_states(N, states=cosphases)

def _experimental_rand_state_paper(N):
    """
        Generates an experimental random pure state based off of
        paper: https://www.sciencedirect.com/science/article/abs/pii/S0010465511002748

    """
    rng = np.random.default_rng()
    # Samples the random 2^N simplex
    s = rng.uniform(size=2**N)
    s /= np.sum(s)

    a = s**(0.5)
    # minus one to the size since we will insert unity at the front
    p = np.exp(rng.uniform(high=2*np.pi, size=2**N - 1) * 1j)
    p = np.insert(p, 0, 1)

    return qstate_imager.qobj_from_array(a * p)

def _experimental_rand_state(N):
    """
        Experimental approach to generating a random haar state.
    
        Args:
            N (int): the number of qubits in the system this state will
            represent.
    
        Returns:
            q (qutip.Qobj): the state    
    """
    components = np.array([complex(np.random.normal(), np.random.normal()) for _ in range(2**N)])

    # Normalize the state
    components /= np.sqrt(np.sum(np.abs(components)**2))

    return qstate_imager.qobj_from_array(components)

def save_rand_states_unitary_method(N, nsamples=1, percent=0.2, root='results'):
    """
        Generates random pure Haar states by generating a random unitary
        Haar matrix and pulling states from the columns. Has the caveat that
        all states generated from a single matrix will be related to each
        other by orthonormality. To combat this we request the percent of
        states to be used from the unitary matrix. To only grab a subset of
        all of the columns before regenerating a new unitary matrix.
    
        Args:
            N (int): the number of qubits in the system
            nsamples (int): the number of random Haar states to generate
            percent (float): the percentage of columns to use from the unitary
                matrix.
            root (str): the base file path used for general saving
    
        Returns:
            none (None): none
    
    """
    if N > 14:
        print('RAM usage for generating states larger than to N=14 far exceeds 16GB. Exiting...')
        sys.exit()

    #------------- Saving preparation -------------#

    save_flags = {
        'save_post_circuit': False,
        'training': False,
        'entropy': False,
        'save_state': True,
    }

    _prepare_saving(N, 'rand', '', N//2, 'Universal', save_path=root, save_flags=save_flags)

    # Gives the number of samples we collect per unitary matrix generated
        # If we can gather all of our data from a single unitary, then
        # just set it to the number of samples
    samples_per_unitary = min(int( np.ceil(2**N * percent) ), nsamples)
    num_unitary_to_generate = int( np.ceil( nsamples / samples_per_unitary ) )

    bar = ShadyBar('Random Haar generation progress', max=nsamples, suffix='%(percent)d%%')
    
    count = 0
    for i in range(num_unitary_to_generate):
        U = qt.rand_unitary_haar(2**N)

        for j in range(samples_per_unitary):

            # We should be on the last unitary matrix and have all the data
                # we need so we can break out
            if count >= nsamples:
                break

            path = get_save_state_file_path(N, 'rand', '', N//2, type_of_circuit='Universal', trial=count, root=root)
            # Save the state
            qstate_imager.save_state(U[:, j], path)

            count += 1
            bar.next()

    bar.finish()

######################## Random gates ########################
def rand_cnot(N):
    """
        Create a random CNOT gate. The CNOT gate takes an input gate (control)
        and depending on the control, changes the value of the target gate. It
        performs the NOT operation on the target qubit iff the control qubit
        is |1>, otherwise, it leaves the target qubit untouched.
    
        Args:
            N (int): dimension of your gate, so you can create an unitary 
            matrix of appropriate size
    
        Returns:
             (3-tuple): (cnot gate, control, target)
    
    """
    [control, target] = random.sample(range(N), 2)
    # Changes the target the gate is applied to if the local flag is
        # passed.
    if _ring_topology:
        # The topology of the qubits is assumed to be a ring to ensure
            # endpoints have an equal probability of being selected as other
            # qubits. So if the control already picks the last qubit, we move
            # the control to the first qubit.
        target = control + 1 if control + 1 < N else 0
    return (qt.qip.operations.cnot(N, control, target), control, target)

def rand_hadamard(N):
    """
        Generates a Hadamard gate that acts on a single qubit. If the input is 
        |0>, it maps  to (|0> + |1>)/sqrt(2). If the input is |1> it maps to
        (|0> - |1>)/sqrt(2). This is effectively a rotation within the basis
        |0>, |1>.
    
        Args:
            N (int): dimension of your gate, so you can create an unitary 
            matrix of appropriate size
    
        Returns:
             (2-tuple): unitary matrix representing SNOT gate applied to target
    
    """
    [target] = random.sample(range(N), 1)
    return (qt.qip.operations.snot(N, target), target)

def rand_phasegate(N, phase=random.uniform(0, 2*np.pi)):
    """
        Generates a phase gate where both the phase can be made randomly or 
        fixed (as in the case of an S or T gate), and its target is random.
    
        Args:
            N (int): number of qubits in the system. Used to scale the 
            phasegate accordingly
            phase (float): the phase for the gate
    
        Returns:
            phasegate (2-tuple): ordered tuple containing gate and target
    
    """
    [target] = random.sample(range(N), 1)
    return (qt.qip.operations.phasegate(phase, N, target), target)

def rand_S(N):
    """
        Generates a phase gate that acts on a single qubit. It changes the phase by pi/2 but does not change the probability of the state being |0> or |1>. This phase change belongs to Clifford gate set. 
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size
    
        Returns:
             (2-tuple): unitary matrix representing phase gate applied to control and target
    
    """
    return rand_phasegate(N, phase=np.pi/2)

def rand_T(N):
    """
        Generates a phase gate that acts on a single qubit. It changes the phase by pi/4 but does not change the probability of the state being |0> or |1>. This phase change corresponds to Universal gate set. 
    
        Args:
            N (int): dimension of your gate, so you can create an unitary matrix of appropriate size
    
        Returns:
             (3-tuple): unitary matrix representing PHASE gate applied to control and target
    
    """
    return rand_phasegate(N, phase=np.pi/4)

########################Circuits########################
def rand_circuit(Gates, N, d):
    """
        Generates a random circuit represented as a list of gates to be successively applied to some input quantum state.
    
        Args:
            Gates (arg1 type): the gates available for applying to the circuit
            N (int): the number of qubits. Also related to the dimension of the system
            d (int): the number of gates to add to the circuit
    
        Returns:
            circuit (list): the list of gates ordered by when they should be applied to the states
    
    """
    gates = random.choices(Gates, k=d)
    # For each gate, we must take it, expand it to our N-qbit system
        # Qutip then returns (gate, N, control, target) and we must pull out
        # the gate itself and append it to our circuit
    return [gate(N)[0] for gate in gates]

def random_circuit_from_instructions(instructions, N):
    """
        Function that runs through instructions to get the
        random quantum circuit.
    
    """
    # Hadamard: snot(N, target)
    # cnot: cnot(N, control, target)
    # S(\pi/4): phasegate(\pi/2, N, target)
    # T: phasegate(\pi/4, N, target)
    circuits = []

    if type(instructions) is str:
        instructions = parse_instructions(instructions)

    for i in range(len(instructions)):
    
        instruction = instructions[i]

        amount_of_gates = instruction[1]
        if instruction[0] == 'Cl':
            gates = [rand_hadamard, rand_cnot, rand_S]
        elif instruction[0] == 'U':
            gates = [rand_hadamard, rand_cnot, rand_T]
        elif instruction[0] == 'T':
            gates = [rand_T]
        else:
            print(f'Invalid circuit type instructions. Type given: {instructions[0]}.')
            print('Exiting...')
            sys.exit()

        circuits.append(rand_circuit(gates, N, amount_of_gates))

    return circuits

def rand_unitary_haar(N):
    """
        Function that generates a random unitary Haar matrix on N qubits.
    
        Args:
            N (int): the number of qubits
    
        Returns:
            U (numpy.ndarray): the random unitary haar matrix
    
    """
    return qt.rand_unitary_haar(2**N).full()

def rand_unitary_clifford(N, circuit_length=1):
    """
        Generates a unitary Clifford matrix which is the matrix method of 
        representing a Clifford circuit.
    
        Args:
            N (int): the number of qubits input into the circuit. Affects the dimensionality of the matrix.
            circuit_length (int): the number of Clifford gates that should be included in the circuit.
    
        Returns:
            U (numpy.ndarray): the unitary matrix
    
    """

    # random_circuit_from_instructions returns a list containing
        # blocks of circuits, since we only wanted Clifford gates we
        # will only have, and will only be interested in the first block
    # Finally, we reverse the circuit, since the gates applied first are the
        # first gates in the circuit, however in matrix multiplication
        # they will be the rightmost unitary operators in the product
        # hence we must multiply them last if we multiply the operators
        # from left to right
    circuit = random_circuit_from_instructions(
        parse_instructions(f'Clx{circuit_length}'), N
    )[0][::-1]

    # Converts the qutip.Qobj to a proper numpy.ndarray
    if circuit_length == 1:
        return circuit[0].full()

    U = circuit[0]

    for gate in circuit[1:]:
        U = np.matmul(U, gate)

    return U

def eval_circuit(circuit, q, gather_entropies=False, bipartition=None, renyi_parameter=1, verbose=False):
    """
        Function that evaluates a quantum circuit by taking in an initial quantum state and returning the output of the circuit
    
        Args:
            circuit (list): the quantum circuit
            q (qObj): the initial input state (i.e. |0>)
            gather_entropies (bool): the option to calculate and store the entropy of the state with each gate application
            bipartition (int): the bipartition to use for the Renyi entropy. 
                Pass in 0 to return the bipartition average of the entropy
            renyi_parameter (float): which Renyi entropy to calculate. Ignored 
                if gather_entropies is false
            verbose (bool): the option to print the output state as each gate 
                is applied
    
        Returns:
            q (qObj): the final quantum state resulting from the evaluation of the circuit
    
    """
    if gather_entropies:
        entropies = []

    for gate in circuit:
        # Sequentially apply each gate in the circuit to each state
        q = gate(q)
        
        if gather_entropies:
            # Calculate the bipartition average
            if bipartition == 0:
                N = int(np.log2(q.shape[0]))

                entropy = np.mean([
                    renyi_entropy(q, bipart, renyi_parameter)
                    for bipart in range(1, N)
                ])
            else:
                entropy = renyi_entropy(q, bipartition, renyi_parameter)

            entropies.append(entropy)
        if verbose:
            print_qstate(q)

    if gather_entropies:
        return q, entropies

    return q

def rearrange_circuit(Cl_gates, T_gates, instructions):
    """
        Function which takes a pre-made collection of gates and rearranges it 
        according to a set of instructions. Useful for seeing how rearranging 
        gates can affect a circuit.
    
        Args:
            Cl_gates (list): the Clifford gates
            T_gates (list): the T-gates
            instructions (str): the new circuit's instructions
    
        Returns:
            circuit (list): the new circuit
    
    """
    instructions = parse_instructions(instructions)
    circuit = []

    Cl_gates_used = T_gates_used = 0

    for (gate_type, num_gates) in instructions:
        if gate_type == 'Cl':
            circuit += [Cl_gates[Cl_gates_used:Cl_gates_used + num_gates]]
            Cl_gates_used += num_gates
        elif gate_type == 'T':
            circuit += [T_gates[T_gates_used:T_gates_used + num_gates]]
            T_gates_used += num_gates
        else:
            print(f'Invalid gate type: {gate_type}')
            print('Returning circuit as-is.')
            break

    return circuit

########################Operations########################
def get_total_gates(instructions):
    """
        Function that returns the total number of gates used in the total random circuit.
    
        Args:
            instructions (list or string): a list of instructions for each 
                circuit to be generated containing information on the number 
                of gates in each piece In the case of a string, it simply calls
                parse_instructions as a helper first.
    
        Returns:
            total_gates (int): the total number of gates used
    
    """
    if type(instructions) is str:
        instructions = parse_instructions(instructions)

    return sum([instruction[1] for instruction in instructions])

def _sing_values_of_bipart_state(q, bipartition=None):
    """
        Calculates the singular values of a bipartitioned state.
    
        Args:
            q (qutip.Qobj): the state
            bipartition (int): the bipartition to use
    
        Returns:
            svd (numpy.ndarray): the singular values
    
    """
    return np.linalg.svd(qstate_imager.partition_state(q, bipartition), compute_uv=False)

def renyi_entropy(q, bipartition, alpha):
    """
        Calculates the Renyi entropy for given state, bipartition, and parameter alpha. Returns the Von-Neumann entropy if alpha = 1.
    
        Args:
            q (qutip.qObj): the state
            bipartition (int): the bipartition to use
            alpha (float): the parameter for the Renyi entropy.
    
        Returns:
            entropy (float): the Renyi entropy
    
    """
    if alpha == 1:
        return vn_entropy(q, bipartition)

    squared_singular_values = np.square(_sing_values_of_bipart_state(q, bipartition))
    return np.log2((squared_singular_values**alpha).sum()) / (1 - alpha)    

def vn_entropy(q, bipartition):
    """
        Calculates the Von Neumann entropy for a bipartitioned state.

        Args:
            q (qutip.Qobj): the state
            bipartition (int): the bipartition

        Returns:
            vn (float): the entropy
            
    """
    squared_singular_values = np.square(_sing_values_of_bipart_state(q, bipartition))
    # remove any zeros to avoid issues when applying logarithm
        # fine since we define 0 log(0) to be 0 anyways
    probabilities = squared_singular_values[squared_singular_values != 0]  
    return -(probabilities * np.log2(probabilities)).sum()

########################Printers########################
def print_matrix(A, precision=4):
    """
    Print a matrix. This implementation currently does not accept a matrix with complex-valued entries.
    """
    (m, n) = A.shape
    print('Matrix of dimension: {0}'.format((m,n)))
    for i in range(m):
        for j in range(n):
            print("{:0.{precision}f}".format(A[i, j], precision=precision),  end=' ')
        print()

def print_qstate(q):
    print(f'Quantum state of dimension: {q.shape}')

    for c, component in enumerate(q):
        print(f'Component: {c+1} > {np.around(component, 4)}')

######################## Saving ########################

def get_serial():
    """
        Generates a serial code for each file to be saved to prevent issues
        with overwriting data.
    
        Args:
            none (None): none
    
        Returns:
            serial (str): the serial
    
    """
    return datetime.now().strftime("%d%m%y%H%M%S")

def get_save_path(N, input_state, instructions, bipartition, local_top=False, root='results'):
    """
        Function that handles path generation based off the signature of the 
        experiment for future saving of data and other files.
    
        Args:
            N (int): the number of qubits in the system
            input_state (string): the input state's label
            instructions (string): the string encoding the instructions
                of the circuit used
            bipartition (int): the type of bipartition used.
            local_top (bool): where the qubits are arranged in a ring 
                topologically
            root (string): the base location for all of the results to be
                saved to.
    
        Returns:
            path (string): the location to save to
    
    """
    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'
    if local_top:
        fname_template += '-local_top'

    if bipartition != N//2:
        fname_template += f'-bipart({bipartition})'

    return os.path.join(root, fname_template)

def get_training_data_path(N, input_state, instructions, bipartition, type_of_circuit, local_top=False, root='results'):
    """
        Function that handles path generation for training data folder based 
        off the signature of the experiment
    
        Args:
            N (int): the number of qubits in the system
            input_state (string): the input state
            instructions (string): the string encoding the instructions
                of the circuit used
            bipartition (int): the type of bipartition used
            type_of_circuit (string): the type of the circuit as a string, 
                i.e. Clifford or Universal for easy identification by
                the machine learning classifier.
            local_top (bool): where the qubits are arranged in a ring 
                topologically
            root (string): the base location for all of the results to be
                saved to.
    
        Returns:
            path (string): the location to save to
    
    """
    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'
    if local_top:
        fname_template += '-local_top'

    if bipartition != N//2:
        fname_template += f'-bipart({bipartition})'

    save_path = get_save_path(N, input_state, instructions, bipartition, local_top=local_top, root=root)
    return os.path.join(save_path, 'training_data-' + fname_template + '-' + type_of_circuit)

def get_training_data_file_path(N, input_state, instructions, bipartition, type_of_circuit, trial, local_top=False, root='results'):
    """
        Function that handles path generation for the individual training data 
        file. Intended for creating consistent file names for training data
        files to be read by the machine learning classifier.
    
        Args:
            N (int): the number of qubits in the system
            input_state (string): the input state
            instructions (string): the string encoding the instructions
                of the circuit used
            bipartition (int): the type of bipartition used
            type_of_circuit (string): the type of the circuit as a string, 
                i.e. Clifford or Universal for easy identification by
                the machine learning classifier.
            trial (int): the current trial we are on, important for enumerating
                the training data files.
            local_top (bool): where the qubits are arranged in a ring 
                topologically
            root (string): the base location for all of the results to be
                saved to.
    
        Returns:
            path (string): the location to save to
    
    """
    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'
    if local_top:
        fname_template += '-local_top'

    if bipartition != N//2:
        fname_template += f'-bipart({bipartition})'

    training_data_path = get_training_data_path(N, input_state, instructions, bipartition, type_of_circuit, local_top=local_top, root=root)

    return os.path.join(
        training_data_path,
        fname_template + f'-{type_of_circuit}-trial({trial})'
    )

def get_save_state_path(N, input_state, instructions, bipartition, type_of_circuit, local_top=False, root='results'):
    """
        Returns the path to the folder to save state files.
    
        Returns:
            path (str): the path
    
    """
    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'

    save_path = get_save_path(N, input_state, instructions, bipartition, local_top=local_top, root=root)
    return os.path.join(save_path, 'states-' + fname_template + '-' + type_of_circuit)

def get_save_state_file_path(N, input_state, instructions, bipartition, type_of_circuit, trial, local_top=False, root='results'):
    """
        Returns the path to save each file. Distinct from get_save_state_path() which returns the path to the folder not the filename.
    
        Returns:
            path (str): the path
    
    """
    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'

    save_state_path = get_save_state_path(N, input_state, instructions, bipartition, type_of_circuit, local_top, root)

    return os.path.join(
        save_state_path,
        'state-' + fname_template + f'-{type_of_circuit}-trial({trial})-{get_serial()}'
    )

def get_entropy_file_path(N, input_state, instructions, bipartition, renyi_parameter=1, local_top=False, root='results'):
    """
        Function that handles path generation for saving entropy data based on
        the experiment signature.
    
        Args:
            N (int): the number of qubits in the system
            input_state (string): the input state
            instructions (string): the string encoding the instructions
                of the circuit used
            bipartition (int): the type of bipartition used
            renyi_parameter (float): the parameter determining which Renyi entropy to use
            local_top (bool): where the qubits are arranged in a ring 
                topologically
            root (string): the base location for all of the results to be
                saved to.
    
        Returns:
            path (string): the location to save to
    
    """

    fname_template = f'N({N})-q0({input_state})-inst[{instructions}]'
    if local_top:
        fname_template += '-local_top'

    if bipartition != N//2:
        fname_template += f'-bipart({bipartition})'

    save_path = get_save_path(N, input_state, instructions, bipartition, local_top=local_top, root=root)
    file_suffix = '-entropies.txt' if renyi_parameter == 1 else f'-entropies({renyi_parameter}).txt'
    return os.path.join(save_path, fname_template + file_suffix)

def save_matrix(fname, matrix, compressed=False, overwrite=False):
    """
        Save a matrix to a text file unless compressed flag is passed in which case the matrix is saved in the .npz format. Defaults to appending data to the file rather than overwriting it.
    
    """
    if not compressed:
        with open(fname, 'a' if not overwrite else 'w') as f:
            np.savetxt(f, matrix)
    else:
        scipy.sparse.save_npz(fname, scipy.sparse.csc_matrix(matrix))

def read_info_file(folder_path, file_name='info'):
    """
        Function for reading and parsing data from info.txt file found
        in simulation folder containing additional information such as the
        total number of trials done.
    
        Args:
            folder_path (string): the location of the folder itself, the name 
                of the file is handled by the script.
    
        Returns:
            info (dict): a dictionary containing any relevant information from
                the file
    
    """
    info = {}
    
    try:
        with open(os.path.join(folder_path, file_name + '.txt')) as f:
            info = ast.literal_eval(f.read())
    except IOError:
        # If the file does not exist that is fine, it will be created on
            # update
        pass

    return info

def update_info_file(folder_path, new_info, file_name='info'):
    """
        Function that takes in information to be added to the information
        file to the corresponding experiment. Information can include things
        like the number of samples taken already, and seeds for random states
        used.
    
        Args:
            folder_path (string): the file path for the info file located in 
                the base of the experiment folder.
            info (dict): a dictionary containing the key and value pairs of
                the information to be written.
    
        Returns:
            none (none): none
    
    """
    # read the old info file and update its contents with this new information
    info = read_info_file(folder_path, file_name)
    info.update(new_info)

    with open(os.path.join(folder_path, file_name + '.txt'), 'w') as f:
        f.write(str(info))

######################## Main body ########################
def randomqc_varied_doping(N=8, U1='Clx640', U2_Ts=[0, 1, 8], U3_Cls=[100], q0_label='zero', q0=None, renyi_parameter=1, save_post_circuit_flag=False, save_state=False, save_path='results/'):
    """
        Variation of randomqc which fixes the entire circuit and the input 
        state but changes ONLY the number of T gates doped into the circuit 
        and the number of Clifford gates in the final block.
    
        Args:
            N (int): the number of qubits in the system
            U1 (str): the instructions for the first circuit block
            U2_Ts (list): a list of integers determining the amount of T gates 
                we want to dope the circuit with. i.e., U2_Ts=[0, 1, 8] will 
                fix the rest of the circuit and first dope with 0 T-gates, 
                then 1 T-gate, then 8 T-gates.
            U2_Cls (list): a list of integers determining the number of 
                Clifford gates for the final block as it is varied similar to 
                U2_Ts.
            q0_label (str): the label for the input state
            q0 (qutip.Qobj) the input state to be fixed
            renyi_parameter (float): the type of Renyi entropy to calculate.
            save_path (str): the base location to save the data
    
        Returns:
            none (None): none
    
    """

    # Generates the complete circuit with the highest number of doped T-gates
        # and longest Clifford gate
    total_circuit = random_circuit_from_instructions(parse_instructions(f'{U1};Tx{max(U2_Ts)};Clx{max(U3_Cls)}'), N)

    for n_T in U2_Ts:
        for n_Cl in U3_Cls:

            # The circuit label for this doping trial
            circuit_label = f'{U1};Tx{n_T};Clx{n_Cl}'

            # Take the first circuit block, only the amount of T-gates needed 
                # for this doping, then the final circuit block only the 
                # amount of Clifford gates needed, that is this current circuit
            circuit = [
                total_circuit[0],
                total_circuit[1][:n_T],
                total_circuit[2][:n_Cl]
            ]

            # Use the non-random qc since we've handled the randomization outside
                # of the call here
            qc(N, circuit_label, circuit, q0_label, q0, entropy_flag=True, renyi_parameter=renyi_parameter, save_post_circuit_flag=save_post_circuit_flag, save_state=save_state, save_path=save_path)

def qc(
        N, circuit_label, circuit, q0_label, q0,
        bipartition=None, ring_topology=False,
        verbose_flag=False, save_post_circuit_flag=False, training_flag=False, entropy_flag=False, save_state=False, save_path='results/', colored=False, renyi_parameter=1
    ):
    """
        Non-random version of randomqc. Operates by taking in a quantum state 
        and a circuit that is generated before calling the function, from there we do the typical gathering of quantities of interest as the state is passed through the circuit.

        Args:
            N (int): the number of qubits in the system
            circuit_label (str): the type of circuit instructions used. 
                Intended for saving purposes. An example string is 
                'Clx640;Tx10;Clx100'.
            circuit (list): the circuit itself
            q0_label (str): the label for the quantum state, intended for 
                saving purposes.
            q0 (qutip.Qobj): the quantum state to pass through the circuit
            bipartition (int): the way to bipartition the state to generate
                images
            ring_topology (bool): whether the target of 2-qubit gates should
                only be applied to neighboring qubits
            verbose_flag (bool): the option to print out verbosely
            save_post_circuit_flag (bool): the option to save images after each
                block of the circuit is applied
            training_flag (bool): the option to gather training data for
                classification systems
            entropy_flag (bool): the option to gather entropy data
            save_state (bool): the option to save the output state of the
                entire circuit as a qutip file
            save_path (str): the base location for data to be saved to
            colored (bool): the option to add a colormap to images post circuit
                blocks. Does NOT apply to training images
            renyi_parameter (float): the type of Renyi entropy to gather
    
        Returns:
            q (qutip.Qobj): the output state of the circuit            

    """

   # The root of the folder to be saved to
    root = save_path
    # Check N
    if N < 2:
        print('Invalid number of qubits to system. Must be at least 2 in order to use cnot gate.')
        print('Exiting...')
        sys.exit()

    # Check bipartition
    if bipartition is None:
        # Split the qubits in half, use integer division to floor it in the 
            # case of an odd number of qubits.
        bipartition = N//2
    elif bipartition < 0 or bipartition >= N:
        print('Invalid bipartition. Must be an integer greater than or equal to 0 and less than the number of qubits in the system.')
        print('Exiting...')
        sys.exit()

    #------------- Initialization -------------#

    # To work with legacy code
    instructions_string = circuit_label
    instructions = parse_instructions(instructions_string)

    # Adjust any capitalization on input state
    q0_label = q0_label.lower()

    # Check to ensure there are any T or Universal gates in the circuit
        # making the circuit overall Universal. Otherwise it must
        # only contain Clifford gates
    type_of_circuit = 'Universal' if 'rand' in q0_label or 'T' in instructions_string or 'U' in instructions_string else 'Clifford'

    global _ring_topology
    _ring_topology = ring_topology

    save_flags = {
        'save_post_circuit': save_post_circuit_flag,
        'training': training_flag,
        'entropy': entropy_flag,
        'save_state': save_state,
    }

    # If we are taking the bipartition average for the entropy data
        # then we won't be able to gather images
    if bipartition == 0:
        if save_flags['save_post_circuit'] or save_flags['training']:
            print('Cannot save images when calculating the bipartition average. Turning off save image flags...')
            save_flags['save_post_circuit'] = save_flags['training'] = False


    # If all of our flags are false, we don't want to save anything
    no_save_flag = not True in save_flags.values()
    # If all of our flags are true, then we are saving everything
    save_all_flag = not False in save_flags.values()
    # Save these in our flags
    save_flags['no_save'] = no_save_flag
    save_flags['save_all'] = save_all_flag

    #------------- Check flags -------------#

    # Since we're not saving nothing, we want to have a directory ready
        # to hold everything
    if not save_flags['no_save']:
        _prepare_saving(N, q0_label, instructions_string, bipartition, type_of_circuit, save_path, save_flags)

        save_path = get_save_path(N, q0_label, instructions_string, bipartition, local_top=_ring_topology, root=root)

        if save_flags['training']:
            training_data_path = get_training_data_path(N, q0_label, instructions_string, bipartition, type_of_circuit, local_top=_ring_topology, root=root)

    # Print formatting heading the simulation
    if verbose_flag:
        print(f'System of {N} qubits, Input State: {q0_label}, Topology: {"ring" if ring_topology else "totally connected graph"}')
        print()
    else:
        # We don't want a progress bar in the case where we're already 
            # printing out other statements indicating progress
        bar = ShadyBar('Circuit progress', max=(get_total_gates(instructions_string) + 1), suffix='%(percent)d%%')

    # Sample counter in case of doing additional trials 
        # i.e. if we've done 100 trials before, and want to collect data for
        # 100 more trials, we want the sample counter to find out that it
        # is really 101/200 instead of 1/100
    if save_flags['save_post_circuit'] or save_flags['training'] or save_flags['save_state']:
        info = read_info_file(save_path)
        # Check if we already have any samples recorded else start at 0
        sample_count = info['total_samples'] if 'total_samples' in info else 0

    #------------- Trial entry -------------#

    # Reset the input state through the label passed or otherwise
        # copy the fixed state to reuse it as an input
    q = q0.copy()

    if save_flags['entropy']:
        # Vector that holds the current trial's entropy data
        if bipartition == 0:
            initial_entropy = np.mean([
                renyi_entropy(q, bipart, renyi_parameter)
                for bipart in range(1, N)
            ])               
        else:
            initial_entropy = renyi_entropy(q, bipartition, renyi_parameter)

        entropy_data = np.array([initial_entropy])

    # We update the name of the file for this specific trial using the
        # template generated from the arguments
    if save_flags['save_post_circuit']:
        trial_folder = os.path.join(save_path, f'trial({sample_count})')

        # If the folder already exists, but the info file is telling us
            # that the number of samples collected is less than the trial
            # folder already existing, then this folder may be from an
            # interrupted simulation. In which case we don't trust this 
            # data anyways and are okay with overwriting it.
        os.makedirs(trial_folder, exist_ok=True)
        # Saves the input state before circuit evaluation
        qstate_imager.save_state_as_image(
            q,
            colored,
            os.path.join(trial_folder, f'q0({q0_label})'),
            bipartition=bipartition
        )

    # Now that we've generated the image of the initial state, we are
        # ready to update our progress
    if not verbose_flag:
        bar.next()

    #------------- Circuit creation -------------#

    # To work with legacy code
    circuits = circuit

    for i, circuit in enumerate(circuits):
        
        instruction = instructions[i]
        amount_of_gates = instruction[1]

        #------------- Circuit evaluation -------------#

        if save_flags['entropy']:
            q, circuit_entropy_data = eval_circuit(circuit, q, gather_entropies=True, bipartition=bipartition, renyi_parameter=renyi_parameter)
            entropy_data = np.append(entropy_data, circuit_entropy_data)
        else:
            q = eval_circuit(circuit, q, gather_entropies=False)

        #------------- Printing -------------#

        # Perform any desired verbose behavior here such as
            # printing of the state or saving additional files
        if verbose_flag:
            print(f'Circuit number: {i + 1}, Type of Gates: {instruction[0]}, Number of Gates: {amount_of_gates}')
            print('L2 normed entries of density matrix')
            print_matrix(qstate_imager.state_to_matrix(q, bipartition))
            print()
            # Print the norm of the output state. Expected to be 1
            print(f'Norm of output state: {q.norm(norm="l2")}')
            ## print out the von Neumann entropy of the pure state ##
            print(f'Renyi entropy of output state: {renyi_entropy(q, bipartition, renyi_parameter)}')
            print(f'With Renyi parameter {renyi_parameter}')
            print(f'With bipartition: {bipartition}')
            print()

        #------------- Saving of data -------------#

        if save_flags['save_post_circuit']:
            # Save the state after this piece of the circuit
            qstate_imager.save_state_as_image(q, colored,
                os.path.join(
                    trial_folder,
                    f'C{i + 1}({instruction[0]}x{amount_of_gates})-trial({sample_count})'
                ),
                bipartition=bipartition
            )

        if not verbose_flag:
            bar.next()

        #------------- Post circuit evaluation -------------#
        
        # Saves the output state as a readable qObj for later manipulations
        if save_flags['save_state']:
            save_state_file_path = get_save_state_file_path(N, q0_label, instructions_string, bipartition, type_of_circuit, sample_count, local_top=_ring_topology, root=root)
            qstate_imager.save_state(q, save_state_file_path)

        if save_flags['training']:
            # Save the final output of the circuit for the training data
            # Example training data file names
                # N(12)-q0(rand)-inst[circuithere]-Universal-trial(3).jpeg
                # N(12)-q0(prod)-inst[circuithere]-Clifford-trial(3).jpeg
            training_data_file_path = get_training_data_file_path(N, q0_label, instructions_string, bipartition, type_of_circuit, sample_count, local_top=_ring_topology, root=root)
            
            ### Change line for any change of basis on output states ###
            # q = qstate_imager.qubit_change_of_basis(q, basis='x')
            
            qstate_imager.save_state_as_image(q, False, training_data_file_path, bipartition=bipartition)

        # Increments the total amount of samples done in the case where
            # we are saving either circuit data or training data
        if save_flags['save_post_circuit'] or save_flags['training'] or save_flags['save_state']:
            sample_count += 1
            update_info_file(
                save_path,
                {'total_samples': sample_count},
            )

        # Save the entropy data to a file
        if save_flags['entropy']:
            entropy_file_path = get_entropy_file_path(N, q0_label, instructions_string, bipartition, renyi_parameter=renyi_parameter, local_top=_ring_topology, root=root)

            # Convert to the proper row vector format for a single trial
            entropy_data = entropy_data.reshape(1, -1)
            save_matrix(entropy_file_path, entropy_data)

    #------------- Post simulation -------------#
    if not verbose_flag:
        bar.finish()

    return q

def randomqc(
        N=8, instructions='Clx100', q0_label='zero', q0=None,
        samples=1, bipartition=None, ring_topology=False,
        verbose_flag=False, save_post_circuit_flag=False, training_flag=False, entropy_flag=False, save_state=False, save_path='results/', colored=False, renyi_parameter=1
    ):
    """
        The main body of the randqc script. Takes in the necessary parameters
        for a random quantum circuit and allows the circuit to operate on
        an input state one gate at a time with the option to gather data such
        as the entropy of the state versus the gate applied and generate images
        depending on the bipartition passed in.
    
        Args:
            N (int): the number of qubits in the system
            instructions (string): a string formatted to encode the structure
                of the random quantum circuit of interest.
            q0_label (string): a string containing the type of input
                state to use, options are: zero, prod, or rand, for a zero 
                state, product state, or a random Haar state respectively.
            q0 (qutip.Qobj): the input state, leave as None to generate a state
                through the label passed in, or pass in a state to keep it 
                fixed.
            samples (int): the number of trials to run
            bipartition (int): the way to bipartition the state to generate
                images
            ring_topology (bool): whether the target of 2-qubit gates should
                only be applied to neighboring qubits
            verbose_flag (bool): the option to print out verbosely
            save_post_circuit_flag (bool): the option to save images after each
                block of the circuit is applied
            training_flag (bool): the option to gather training data for
                classification systems
            entropy_flag (bool): the option to gather entropy data
            save_state (bool): the option to save the output state of the
                entire circuit as a qutip file
            save_path (str): the base location for data to be saved to
            colored (bool): the option to add a colormap to images post circuit
                blocks. Does NOT apply to training images
            renyi_parameter (float): the type of Renyi entropy to gather
    
        Returns:
            q (qutip.Qobj): the output state of the circuit

    """

    #------------- Error checking of arguments -------------#  

    # The root of the folder to be saved to
    root = save_path
    # Check N
    if N < 2:
        print('Invalid number of qubits to system. Must be at least 2 in order to use cnot gate.')
        print('Exiting...')
        sys.exit()

    # Check bipartition
    if bipartition is None:
        # Split the qubits in half, use integer division to floor it in the 
            # case of an odd number of qubits.
        bipartition = N//2
    elif bipartition < 0 or bipartition >= N:
        print('Invalid bipartition. Must be an integer greater than or equal to 0 and less than the number of qubits in the system.')
        print('Exiting...')
        sys.exit()

    #------------- Initialization -------------#

    # Calls helper function to parse instruction string
    instructions_string = instructions
    instructions = parse_instructions(instructions)

    # Adjust any capitalization on input state
    q0_label = q0_label.lower()

    # Check to ensure there are any T or Universal gates in the circuit
        # making the circuit overall Universal. Otherwise it must
        # only contain Clifford gates
    type_of_circuit = 'Universal' if 'rand' in q0_label or 'T' in instructions_string or 'U' in instructions_string else 'Clifford'

    global _ring_topology
    _ring_topology = ring_topology

    save_flags = {
        'save_post_circuit': save_post_circuit_flag,
        'training': training_flag,
        'entropy': entropy_flag,
        'save_state': save_state,
    }

    # If we are taking the bipartition average for the entropy data
        # then we won't be able to gather images
    if bipartition == 0:
        if save_flags['save_post_circuit'] or save_flags['training']:
            print('Cannot save images when calculating the bipartition average. Turning off save image flags...')
            save_flags['save_post_circuit'] = save_flags['training'] = False


    # If all of our flags are false, we don't want to save anything
    no_save_flag = not True in save_flags.values()
    # If all of our flags are true, then we are saving everything
    save_all_flag = not False in save_flags.values()
    # Save these in our flags
    save_flags['no_save'] = no_save_flag
    save_flags['save_all'] = save_all_flag

    #------------- Check flags -------------#

    # Since we're not saving nothing, we want to have a directory ready
        # to hold everything
    if not save_flags['no_save']:
        _prepare_saving(N, q0_label, instructions_string, bipartition, type_of_circuit, save_path, save_flags)

        save_path = get_save_path(N, q0_label, instructions_string, bipartition, local_top=_ring_topology, root=root)

        if save_flags['training']:
            training_data_path = get_training_data_path(N, q0_label, instructions_string, bipartition, type_of_circuit, local_top=_ring_topology, root=root)

    # Print formatting heading the simulation
    if verbose_flag:
        print(f'System of {N} qubits, Input State: {q0_label}, Topology: {"ring" if ring_topology else "totally connected graph"}')
        print()
    else:
        # We don't want a progress bar in the case where we're already 
            # printing out other statements indicating progress
        bar = ShadyBar('Circuit progress', max=(samples * (len(instructions) + 1)), suffix='%(percent)d%%')

    # Sample counter in case of doing additional trials 
        # i.e. if we've done 100 trials before, and want to collect data for
        # 100 more trials, we want the sample counter to find out that it
        # is really 101/200 instead of 1/100
    if save_flags['save_post_circuit'] or save_flags['training'] or save_flags['save_state']:
        info = read_info_file(save_path)
        # Check if we already have any samples recorded else start at 0
        sample_count = info['total_samples'] if 'total_samples' in info else 0

    #------------- Trial entry -------------#

    for trial in range(samples):
        # Reset the input state through the label passed or otherwise
            # copy the fixed state to reuse it as an input
        q = parse_state_string(N, q0_label) if q0 is None else q0.copy()

        if save_flags['entropy']:
            # Vector that holds the current trial's entropy data
            if bipartition == 0:
                initial_entropy = np.mean([
                    renyi_entropy(q, bipart, renyi_parameter)
                    for bipart in range(1, N)
                ])               
            else:
                initial_entropy = renyi_entropy(q, bipartition, renyi_parameter)

            entropy_data = np.array([initial_entropy])

        # We update the name of the file for this specific trial using the
            # template generated from the arguments
        if save_flags['save_post_circuit']:
            trial_folder = os.path.join(save_path, f'trial({sample_count})')

            # If the folder already exists, but the info file is telling us
                # that the number of samples collected is less than the trial
                # folder already existing, then this folder may be from an
                # interrupted simulation. In which case we don't trust this 
                # data anyways and are okay with overwriting it.
            os.makedirs(trial_folder, exist_ok=True)
            # Saves the input state before circuit evaluation
            qstate_imager.save_state_as_image(
                q,
                colored,
                os.path.join(trial_folder, f'q0({q0_label})'),
                bipartition=bipartition
            )

        # Now that we've generated the image of the initial state, we are
            # ready to update our progress
        if not verbose_flag:
            bar.next()

        #------------- Circuit creation -------------#

        # Generate the random quantum circuit
        circuits = random_circuit_from_instructions(instructions, N)

        for i, circuit in enumerate(circuits):
            
            instruction = instructions[i]
            amount_of_gates = instruction[1]

            #------------- Circuit evaluation -------------#

            if save_flags['entropy']:
                q, circuit_entropy_data = eval_circuit(circuit, q, gather_entropies=True, bipartition=bipartition, renyi_parameter=renyi_parameter)
                entropy_data = np.append(entropy_data, circuit_entropy_data)
            else:
                q = eval_circuit(circuit, q, gather_entropies=False)

            #------------- Printing -------------#

            # Perform any desired verbose behavior here such as
                # printing of the state or saving additional files
            if verbose_flag:
                print(f'Circuit number: {i + 1}, Type of Gates: {instruction[0]}, Number of Gates: {amount_of_gates}')
                print('L2 normed entries of density matrix')
                print_matrix(qstate_imager.state_to_matrix(q, bipartition))
                print()
                # Print the norm of the output state. Expected to be 1
                print(f'Norm of output state: {q.norm(norm="l2")}')
                ## print out the von Neumann entropy of the pure state ##
                print(f'Renyi entropy of output state: {renyi_entropy(q, bipartition, renyi_parameter)}')
                print(f'With Renyi parameter {renyi_parameter}')
                print(f'With bipartition: {bipartition}')
                print()

            #------------- Saving of data -------------#

            if save_flags['save_post_circuit']:
                # Save the state after this piece of the circuit
                qstate_imager.save_state_as_image(q, colored,
                    os.path.join(
                        trial_folder,
                        f'C{i + 1}({instruction[0]}x{amount_of_gates})-trial({sample_count})'
                    ),
                    bipartition=bipartition
                )

            if not verbose_flag:
                bar.next()

        #------------- Post circuit evaluation -------------#
        
        # Saves the output state as a readable qObj for later manipulations
        if save_flags['save_state']:
            save_state_file_path = get_save_state_file_path(N, q0_label, instructions_string, bipartition, type_of_circuit, sample_count, local_top=_ring_topology, root=root)
            qstate_imager.save_state(q, save_state_file_path)

        if save_flags['training']:
            # Save the final output of the circuit for the training data
            # Example training data file names
                # N(12)-q0(rand)-inst[circuithere]-Universal-trial(3).jpeg
                # N(12)-q0(prod)-inst[circuithere]-Clifford-trial(3).jpeg
            training_data_file_path = get_training_data_file_path(N, q0_label, instructions_string, bipartition, type_of_circuit, sample_count, local_top=_ring_topology, root=root)
            
            ### Change line for any change of basis on output states ###
            # q = qstate_imager.qubit_change_of_basis(q, basis='x')
            
            qstate_imager.save_state_as_image(q, False, training_data_file_path, bipartition=bipartition)

        # Increments the total amount of samples done in the case where
            # we are saving either circuit data or training data
        if save_flags['save_post_circuit'] or save_flags['training'] or save_flags['save_state']:
            sample_count += 1
            update_info_file(
                save_path,
                {'total_samples': sample_count},
            )

        # Save the entropy data to a file
        if save_flags['entropy']:
            entropy_file_path = get_entropy_file_path(N, q0_label, instructions_string, bipartition, renyi_parameter=renyi_parameter, local_top=_ring_topology, root=root)

            # Convert to the proper row vector format for a single trial
            entropy_data = entropy_data.reshape(1, -1)
            save_matrix(entropy_file_path, entropy_data)

    #------------- Post simulation -------------#
    if not verbose_flag:
        bar.finish()

    return q

def main():
    print('In randomqc.py')

    # for i in range(8, 14):
    #     save_rand_states_unitary_method(i, 1000, percent=0.3, root='/Users/Otanan/Documents/RandQC/results')
    # return

    # save_rand_states_unitary_method(13, 400, percent=0.3, root='/Users/Otanan/Documents/RandQC/results')
    # return

    # Typical main
    args = _parse_args()
    randomqc(
        N=args.Nqbits,
        instructions=args.instructions, 
        q0_label=args.input_state,
        samples=args.nsamples,
        bipartition=args.bipartition if args.bipartition >= 0 else None,
        ring_topology=args.local,
        verbose_flag=args.verbose,
        save_post_circuit_flag=args.save_flags['save_post_circuit'],
        training_flag=args.save_flags['training'],
        entropy_flag=args.save_flags['entropy'],
        save_state=args.save_flags['save_state'],
        save_path=args.output_path,
        colored=args.colored,
        renyi_parameter=args.entropy
    )

if __name__ == '__main__':
    main()